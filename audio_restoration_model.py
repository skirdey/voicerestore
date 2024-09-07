import sys
sys.path.append('./BigVGAN')

import torch
import torch.nn as nn
import torchaudio
from BigVGAN import bigvgan
from BigVGAN.meldataset import get_mel_spectrogram
from voice_restore import VoiceRestore


device = 'cuda' if torch.cuda.is_available() else 'cpu'
bigvgan_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=True).to(device)
bigvgan_model.remove_weight_norm()
example_input = torch.randn(1, 16000)  # Example input waveform
example_spec = get_mel_spectrogram(example_input, bigvgan_model.h)

class OptimizedAudioRestorationModel(nn.Module):
    def __init__(self):
        super().__init__()

        
        
        # Initialize VoiceRestore
        self.voice_restore = VoiceRestore(sigma=0.1, transformer={
            'dim': 768, 'depth': 20, 'heads': 16, 'dim_head': 64,
            'skip_connect_type': 'concat', 'max_seq_len': 2000,
        }, num_channels=100)
        self.voice_restore.eval()
        self.voice_restore.to(device)


    def forward(self, wav):
        # Convert to Mel-spectrogram
        processed_mel = get_mel_spectrogram(wav, bigvgan_model.h).to(device)
        
        # Restore audio
        restored_mel = self.voice_restore.sample(processed_mel.transpose(1,2), steps=32, cfg_strength=1.0)
        restored_mel = restored_mel.squeeze(0).transpose(0, 1)
        
        # Convert restored mel-spectrogram to waveform
        restored_wav = bigvgan_model(restored_mel.unsqueeze(0))
        
        return restored_wav
    


def load_optimized_model(save_path, optimization_type="jit"):
    """
    Load the optimized model, adjusting key names in the state dict if necessary.
    
    Parameters:
    - save_path: The file path where the optimized model is saved.
    - optimization_type: Type of optimization used ('jit', 'compile', 'state_dict').
    """
    if optimization_type == "jit":
        # Load the TorchScript model
        optimized_model = torch.jit.load(save_path)
        print("Loaded TorchScript model.")
    elif optimization_type == "compile":
        # Load the compiled model (note: you need to recompile it)
        optimized_model = OptimizedAudioRestorationModel()
        state_dict = torch.load(save_path)

        # Remove '_orig_mod.' from keys if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")  # Remove '_orig_mod.' prefix
            new_state_dict[new_key] = v
        
        # Load the modified state_dict into the model
        optimized_model.load_state_dict(new_state_dict, strict=False)
        optimized_model = torch.compile(optimized_model, backend="inductor")
        print("Loaded and recompiled model.")
    elif optimization_type == "state_dict":
        # Load the state_dict model
        optimized_model = OptimizedAudioRestorationModel()
        state_dict = torch.load(save_path)

        # Remove '_orig_mod.' from keys if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")  # Remove '_orig_mod.' prefix
            new_state_dict[new_key] = v

        # Load the modified state_dict into the model
        optimized_model.load_state_dict(new_state_dict, strict=False)
        print("Loaded model state_dict.")
    else:
        raise ValueError("Invalid optimization type. Choose from 'jit', 'compile', or 'state_dict'.")

    return optimized_model




def restore_audio(model, input_path, output_path):
    wav, sr = torchaudio.load(input_path)
    wav = wav.mean(dim=0, keepdim=True) if wav.dim() > 1 else wav  # Convert to mono if stereo
    
    with torch.inference_mode():
        restored_wav = model(wav)
        restored_wav = restored_wav.squeeze(0).cpu()  # Move to CPU after processing
    
    torchaudio.save(output_path, restored_wav, 24000)

# Example usage
if __name__ == "__main__":
    checkpoint_path = "./checkpoints/voice-restore-20d-16h.pt"
    optimized_model_path = "./optimized_audio_restoration_model.pth"
    
    # Load the optimized model
    optimized_model = load_optimized_model(optimized_model_path, optimization_type="state_dict")
    optimized_model.eval()
    optimized_model.to(device)
    
    # Use the model
    input_path = "test_input.wav"
    output_path = "test_output.wav"
    restore_audio(optimized_model, input_path, output_path)