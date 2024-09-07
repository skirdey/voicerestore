import torch
import torch.nn as nn
import torchaudio
from bigvgan import bigvgan
from bigvgan.meldataset import get_mel_spectrogram
from e2_tts import VoiceRestore

class OptimizedAudioRestorationModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initialize VoiceRestore
        self.voice_restore = VoiceRestore(sigma=0.1, transformer={
            'dim': 768, 'depth': 20, 'heads': 16, 'dim_head': 64,
            'skip_connect_type': 'concat', 'max_seq_len': 2000,
        }, num_channels=100)
        
        # Initialize BigVGAN
        self.bigvgan_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)
        self.bigvgan_model.remove_weight_norm()

    def forward(self, wav):
        # Convert to Mel-spectrogram
        processed_mel = get_mel_spectrogram(wav, self.bigvgan_model.h)
        
        # Restore audio
        restored_mel = self.voice_restore.sample(processed_mel.transpose(1,2), steps=32, cfg_strength=1.0)
        restored_mel = restored_mel.squeeze(0).transpose(0, 1)
        
        # Convert restored mel-spectrogram to waveform
        restored_wav = self.bigvgan_model(restored_mel.unsqueeze(0))
        
        return restored_wav.squeeze(0)

def load_model(checkpoint_path):
    model = OptimizedAudioRestorationModel()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def save_optimized_model(model, save_path):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, save_path)

def restore_audio(model, input_path, output_path):
    wav, sr = torchaudio.load(input_path)
    wav = wav.mean(dim=0, keepdim=True) if wav.dim() > 1 else wav  # Convert to mono if stereo
    
    with torch.no_grad():
        restored_wav = model(wav)
    
    torchaudio.save(output_path, restored_wav.unsqueeze(0), 24000)  # BigVGAN outputs at 24kHz

# Example usage
if __name__ == "__main__":
    checkpoint_path = "./checkpoints/voice-restore-20d-16h.pt"
    optimized_model_path = "./optimized_audio_restoration_model.pth"
    
    # Load the original model
    original_model = load_model(checkpoint_path)
    
    # Save the optimized model
    save_optimized_model(original_model, optimized_model_path)
    
    # Load the optimized model
    optimized_model = load_model(optimized_model_path)
    
    # Use the model
    input_path = "path/to/input/audio.wav"
    output_path = "path/to/output/restored_audio.wav"
    restore_audio(optimized_model, input_path, output_path)