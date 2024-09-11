import sys
sys.path.append('./BigVGAN')

import torch
import torch.nn as nn
import torchaudio
from BigVGAN import bigvgan
from BigVGAN.meldataset import get_mel_spectrogram
from voice_restore import VoiceRestore


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# If running on non-windows syste, you can try using cuda kernel for faster processing `use_cuda_kernel=True`
bigvgan_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False).to(device)
bigvgan_model.remove_weight_norm()
example_input = torch.randn(1, 16000)  # Example input waveform
example_spec = get_mel_spectrogram(example_input, bigvgan_model.h)

class OptimizedAudioRestorationModel(nn.Module):
    def __init__(self, target_sample_rate=24000):
        super().__init__()

        # Initialize VoiceRestore
        self.voice_restore = VoiceRestore(sigma=0.0, transformer={
            'dim': 768, 'depth': 20, 'heads': 16, 'dim_head': 64,
            'skip_connect_type': 'concat', 'max_seq_len': 2000,
        }, num_channels=100)
        self.voice_restore.eval()
        self.voice_restore.to(device)
        self.target_sample_rate = target_sample_rate


    def forward(self, audio, steps=32, cfg_strength=1.0):
        # Convert to Mel-spectrogram
        processed_mel = get_mel_spectrogram(audio, bigvgan_model.h).to(device)
    
        # Restore audio
        restored_mel = self.voice_restore.sample(processed_mel.transpose(1,2), steps=steps, cfg_strength=cfg_strength)
        restored_mel = restored_mel.squeeze(0).transpose(0, 1)
        
        # Convert restored mel-spectrogram to waveform
        restored_wav = bigvgan_model(restored_mel.unsqueeze(0))
        
        return restored_wav
    


def load_model(save_path):
    """
    Load the model.
    
    Parameters:
    - save_path: The file path where the optimized model is saved.
    """

    optimized_model = OptimizedAudioRestorationModel()
    state_dict = torch.load(save_path)

    optimized_model.voice_restore.load_state_dict(state_dict, strict=True)

    return optimized_model


def restore_audio(model, input_path, output_path, steps=64, cfg_strength=0.5):  
    audio, sr = torchaudio.load(input_path)

    if sr != model.target_sample_rate:
        audio = torchaudio.functional.resample(audio, sr, model.target_sample_rate)


    audio = audio.mean(dim=0, keepdim=True) if audio.dim() > 1 else audio  # Convert to mono if stereo
    
    with torch.inference_mode():
        restored_wav = model(audio, steps=steps, cfg_strength=cfg_strength)
        restored_wav = restored_wav.squeeze(0).cpu()  # Move to CPU after processing
    
    torchaudio.save(output_path, restored_wav, model.target_sample_rate)

# Example usage
if __name__ == "__main__":
    checkpoint_path = "J:\git\voice-restore\checkpoints\voice-restore-20d-16h.pt"
    
    # Load the optimized model
    optimized_model = load_model(checkpoint_path)
    optimized_model.eval()
    optimized_model.to(device)
    
    # Use the model
    input_path = "test_input.wav"
    output_path = "test_output.wav"
    restore_audio(optimized_model, input_path, output_path)
