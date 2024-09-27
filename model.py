import torch
from BigVGAN.meldataset import get_mel_spectrogram
from voice_restore import VoiceRestore


class OptimizedAudioRestorationModel(torch.nn.Module):
    def __init__(self, target_sample_rate=24000, device=None, bigvgan_model=None):
        super().__init__()

        # Initialize VoiceRestore
        self.voice_restore = VoiceRestore(
            sigma=0.0, 
            transformer={
                'dim': 768, 
                'depth': 20, 
                'heads': 16, 
                'dim_head': 64,
                'skip_connect_type': 'concat', 
                'max_seq_len': 2000,
            }, 
            num_channels=100
        )  
        
        self.device = device
        if self.device == 'cuda':
            self.voice_restore.bfloat16()
        self.voice_restore.eval()
        self.voice_restore.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.bigvgan_model = bigvgan_model
        


    def forward(self, audio, steps=32, cfg_strength=0.5):
        # Convert to Mel-spectrogram

        if self.bigvgan_model is None:
            raise ValueError("BigVGAN model is not provided. Please provide the BigVGAN model.")
        
        if self.device is None:
            raise ValueError("Device is not provided. Please provide the device (cuda, cpu or mps).")

        processed_mel = get_mel_spectrogram(audio, self.bigvgan_model.h).to(self.device)

        # Restore audio
        restored_mel = self.voice_restore.sample(processed_mel.transpose(1, 2), steps=steps, cfg_strength=cfg_strength)
        restored_mel = restored_mel.squeeze(0).transpose(0, 1)
        
        # Convert restored mel-spectrogram to waveform
        restored_wav = self.bigvgan_model(restored_mel.unsqueeze(0))
        
        return restored_wav