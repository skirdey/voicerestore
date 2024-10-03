import sys
sys.path.append('./BigVGAN')

import torch
import torch.nn as nn
import torchaudio
import argparse
from BigVGAN import bigvgan
from BigVGAN.meldataset import get_mel_spectrogram
from model import OptimizedAudioRestorationModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# If running on non-windows system, you can try using cuda kernel for faster processing `use_cuda_kernel=True`
bigvgan_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False).to(device)
bigvgan_model.remove_weight_norm()
example_input = torch.randn(1, 16000)  # Example input waveform
example_spec = get_mel_spectrogram(example_input, bigvgan_model.h)


def load_model(save_path):
    """
    Load the model.
    
    Parameters:
    - save_path: The file path where the optimized model is saved.
    """

    optimized_model = OptimizedAudioRestorationModel(device=device, bigvgan_model=bigvgan_model)

    state_dict = torch.load(save_path, map_location=torch.device(device))
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    optimized_model.voice_restore.load_state_dict(state_dict, strict=True)

    return optimized_model


def restore_audio(model, input_path, output_path, steps=16, cfg_strength=0.5):  
    audio, sr = torchaudio.load(input_path)

    if sr != model.target_sample_rate:
        audio = torchaudio.functional.resample(audio, sr, model.target_sample_rate)

    audio = audio.mean(dim=0, keepdim=True) if audio.dim() > 1 else audio  # Convert to mono if stereo
    
    with torch.inference_mode():
        with torch.autocast(device):
            restored_wav = model(audio, steps=steps, cfg_strength=cfg_strength)
            restored_wav = restored_wav.squeeze(0).float().cpu()  # Move to CPU after processing
    
    torchaudio.save(output_path, restored_wav, model.target_sample_rate)


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Audio restoration using OptimizedAudioRestorationModel")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument('--input', type=str, required=True, help="Path to the input audio file")
    parser.add_argument('--output', type=str, required=True, help="Path to save the restored audio file")
    parser.add_argument('--steps', type=int, default=16, help="Number of sampling steps")
    parser.add_argument('--cfg_strength', type=float, default=0.5, help="CFG strength value")

    # Parse arguments
    args = parser.parse_args()

    # Load the optimized model
    optimized_model = load_model(args.checkpoint)
    optimized_model.eval()
    optimized_model.to(device)

    # Use the model to restore audio
    restore_audio(optimized_model, args.input, args.output, steps=args.steps, cfg_strength=args.cfg_strength)
