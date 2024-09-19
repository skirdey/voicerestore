#
# Code is adapted from https://github.com/lucidrains/e2-tts-pytorch
# 

"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""


from __future__ import annotations
from typing import Dict, Any, Optional
from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList, Sequential, Linear
import torch.nn.functional as F

from torchdiffeq import odeint
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, pack, unpack
from x_transformers import Attention, FeedForward, RMSNorm, AdaptiveRMSNorm
from x_transformers.x_transformers import RotaryEmbedding
from gateloop_transformer import SimpleGateLoopLayer

from tensor_typing import Float

class Identity(Module):
    def forward(self, x, **kwargs):
        return x

class AdaLNZero(Module):
    def __init__(self, dim: int, dim_condition: Optional[int] = None, init_bias_value: float = -2.):
        super().__init__()
        dim_condition = dim_condition or dim
        self.to_gamma = nn.Linear(dim_condition, dim)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.constant_(self.to_gamma.bias, init_bias_value)

    def forward(self, x: torch.Tensor, *, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')
        gamma = self.to_gamma(condition).sigmoid()
        return x * gamma

def exists(v: Any) -> bool:
    return v is not None

def default(v: Any, d: Any) -> Any:
    return v if exists(v) else d

def divisible_by(num: int, den: int) -> bool:
    return (num % den) == 0

class Transformer(Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 8,
        cond_on_time: bool = True,
        skip_connect_type: str = 'concat',
        abs_pos_emb: bool = True,
        max_seq_len: int = 8192,
        heads: int = 8,
        dim_head: int = 64,
        num_gateloop_layers: int = 1,
        dropout: float = 0.1,
        num_registers: int = 32,
        attn_kwargs: Dict[str, Any] = dict(gate_value_heads=True, softclamp_logits=True),
        ff_kwargs: Dict[str, Any] = dict()
    ):
        super().__init__()
        assert divisible_by(depth, 2), 'depth needs to be even'

        self.max_seq_len = max_seq_len
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if abs_pos_emb else None
        self.dim = dim
        self.skip_connect_type = skip_connect_type
        needs_skip_proj = skip_connect_type == 'concat'
        self.depth = depth
        self.layers = ModuleList([])

        self.num_registers = num_registers
        self.registers = nn.Parameter(torch.zeros(num_registers, dim))
        nn.init.normal_(self.registers, std=0.02)

        self.rotary_emb = RotaryEmbedding(dim_head)
        self.cond_on_time = cond_on_time
        rmsnorm_klass = AdaptiveRMSNorm if cond_on_time else RMSNorm
        postbranch_klass = partial(AdaLNZero, dim=dim) if cond_on_time else Identity

        self.time_cond_mlp = Sequential(
            Rearrange('... -> ... 1'),
            Linear(1, dim),
            nn.SiLU()
        ) if cond_on_time else nn.Identity()

        for ind in range(depth):
            is_later_half = ind >= (depth // 2)
            gateloop = SimpleGateLoopLayer(dim=dim)
            attn_norm = rmsnorm_klass(dim)
            attn = Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, **attn_kwargs)
            attn_adaln_zero = postbranch_klass()
            ff_norm = rmsnorm_klass(dim)
            ff = FeedForward(dim=dim, glu=True, dropout=dropout, **ff_kwargs)
            ff_adaln_zero = postbranch_klass()
            skip_proj = Linear(dim * 2, dim, bias=False) if needs_skip_proj and is_later_half else None

            self.layers.append(ModuleList([
                gateloop, skip_proj, attn_norm, attn, attn_adaln_zero,
                ff_norm, ff, ff_adaln_zero
            ]))

        self.final_norm = RMSNorm(dim)

    def forward(
        self,
        x: Float['b n d'],
        times: Optional[Float['b'] | Float['']] = None,
    ) -> torch.Tensor:
        batch, seq_len, device = *x.shape[:2], x.device

        assert not (exists(times) ^ self.cond_on_time), '`times` must be passed in if `cond_on_time` is set to `True` and vice versa'

        norm_kwargs = {}

        if exists(self.abs_pos_emb):
            assert seq_len <= self.max_seq_len, f'{seq_len} exceeds the set `max_seq_len` ({self.max_seq_len}) on Transformer'
            seq = torch.arange(seq_len, device=device)
            x = x + self.abs_pos_emb(seq)

        if exists(times):
            if times.ndim == 0:
                times = repeat(times, ' -> b', b=batch)
            times = self.time_cond_mlp(times)
            norm_kwargs['condition'] = times

        registers = repeat(self.registers, 'r d -> b r d', b=batch)
        x, registers_packed_shape = pack((registers, x), 'b * d')

        rotary_pos_emb = self.rotary_emb.forward_from_seq_len(x.shape[-2])

        skips = []

        for ind, (
            gateloop, maybe_skip_proj, attn_norm, attn, maybe_attn_adaln_zero,
            ff_norm, ff, maybe_ff_adaln_zero
        ) in enumerate(self.layers):
            layer = ind + 1
            is_first_half = layer <= (self.depth // 2)

            if is_first_half:
                skips.append(x)
            else:
                skip = skips.pop()
                if self.skip_connect_type == 'concat':
                    x = torch.cat((x, skip), dim=-1)
                    x = maybe_skip_proj(x)

            x = gateloop(x) + x

            attn_out = attn(attn_norm(x, **norm_kwargs), rotary_pos_emb=rotary_pos_emb)
            x = x + maybe_attn_adaln_zero(attn_out, **norm_kwargs)

            ff_out = ff(ff_norm(x, **norm_kwargs))
            x = x + maybe_ff_adaln_zero(ff_out, **norm_kwargs)

        assert len(skips) == 0

        _, x = unpack(x, registers_packed_shape, 'b * d')

        return self.final_norm(x)

class VoiceRestore(nn.Module):
    def __init__(
        self,
        sigma: float = 0.0,
        transformer: Optional[Dict[str, Any]] = None,
        odeint_kwargs: Optional[Dict[str, Any]] = None,
        num_channels: int = 100,
    ):
        super().__init__()
        self.sigma = sigma
        self.num_channels = num_channels

        self.transformer = Transformer(**transformer, cond_on_time=True)

        self.odeint_kwargs = odeint_kwargs or {'atol': 1e-5, 'rtol': 1e-5, 'method': 'midpoint'}

        self.proj_in = nn.Linear(num_channels, self.transformer.dim)
        self.cond_proj = nn.Linear(num_channels, self.transformer.dim)
        self.to_pred = nn.Linear(self.transformer.dim, num_channels)

    def transformer_with_pred_head(self, x: torch.Tensor, times: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.proj_in(x)
        if cond is not None:
            cond_proj = self.cond_proj(cond)
            x = x + cond_proj
        attended = self.transformer(x, times=times)
        return self.to_pred(attended)

    def cfg_transformer_with_pred_head(
        self,
        *args,
        cond=None,
        mask=None,
        cfg_strength: float = 0.5,
        **kwargs,
    ):
        pred = self.transformer_with_pred_head(*args, **kwargs, cond=cond)

        if cfg_strength < 1e-5:
            return pred * mask.unsqueeze(-1) if mask is not None else pred
    
        null_pred = self.transformer_with_pred_head(*args, **kwargs, cond=None)
        
        result = pred + (pred - null_pred) * cfg_strength
        return result * mask.unsqueeze(-1) if mask is not None else result


    @torch.no_grad()
    def sample(self, processed: torch.Tensor, steps: int = 32, cfg_strength: float = 0.5) -> torch.Tensor:
        self.eval()
        epsilon = 1e-5
        times = torch.linspace(epsilon, 1 - epsilon, steps, device=processed.device)


        def ode_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self.cfg_transformer_with_pred_head(x, times=t, cond=processed, cfg_strength=cfg_strength)

        y0 = torch.randn_like(processed)
        trajectory = odeint(ode_fn, y0, times, **self.odeint_kwargs)
        restored = trajectory[-1]
        return restored