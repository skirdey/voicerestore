import sys
sys.path.append('./BigVGAN')

import time
import torch
import torchaudio
import argparse
from tqdm import tqdm
import librosa
from BigVGAN import bigvgan
from BigVGAN.meldataset import get_mel_spectrogram
from model import OptimizedAudioRestorationModel


# Set the device handle macbooks with M1 chip

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize BigVGAN model
bigvgan_model = bigvgan.BigVGAN.from_pretrained(
    'nvidia/bigvgan_v2_24khz_100band_256x',
    use_cuda_kernel=False,
    force_download=False
).to(device)
bigvgan_model.remove_weight_norm()

def measure_gpu_memory():
    if device == 'cuda':
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    return 0



def apply_overlap_windowing_waveform(waveform, window_size_samples, overlap):
    step_size = int(window_size_samples * (1 - overlap))
    num_chunks = (waveform.shape[-1] - window_size_samples) // step_size + 1
    windows = []

    for i in range(num_chunks):
        start_idx = i * step_size
        end_idx = start_idx + window_size_samples
        chunk = waveform[..., start_idx:end_idx]
        windows.append(chunk)
    
    return torch.stack(windows)

def reconstruct_waveform_from_windows(windows, window_size_samples, overlap):
    step_size = int(window_size_samples * (1 - overlap))
    shape = windows.shape
    if len(shape) == 2:
        # windows.shape == (num_windows, window_len)
        num_windows, window_len = shape
        channels = 1
        windows = windows.unsqueeze(1)  # Now windows.shape == (num_windows, 1, window_len)
    elif len(shape) == 3:
        num_windows, channels, window_len = shape
    else:
        raise ValueError(f"Unexpected windows.shape: {windows.shape}")

    output_length = (num_windows - 1) * step_size + window_size_samples

    reconstructed = torch.zeros((channels, output_length))
    window_sums = torch.zeros((channels, output_length))

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_len
        reconstructed[:, start_idx:end_idx] += windows[i]
        window_sums[:, start_idx:end_idx] += 1
    
    reconstructed = reconstructed / window_sums.clamp(min=1e-6)
    if channels == 1:
        reconstructed = reconstructed.squeeze(0)  # Remove channel dimension if single channel
    return reconstructed

def load_model(save_path):
    """
    Load the optimized audio restoration model.
    
    Parameters:
    - save_path: Path to the checkpoint file.
    """
    optimized_model = OptimizedAudioRestorationModel(device=device, bigvgan_model=bigvgan_model)
    state_dict = torch.load(save_path, map_location=device)

    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    optimized_model.voice_restore.load_state_dict(state_dict, strict=True)

    return optimized_model

def restore_audio(model, input_path, output_path, steps=16, cfg_strength=0.5, window_size_sec=5.0, overlap=0.5):  
    # Load the audio file
    start_time = time.time()
    initial_gpu_memory = measure_gpu_memory()
    wav, sr = librosa.load(input_path, sr=24000, mono=True)
    wav = torch.FloatTensor(wav).unsqueeze(0)  # Shape: [1, num_samples]

    window_size_samples = int(window_size_sec * sr)
    step_size = int(window_size_samples * (1 - overlap))

    # Apply overlapping windowing to the waveform
    wav_windows = apply_overlap_windowing_waveform(wav, window_size_samples, overlap)

    restored_wav_windows = []

    for wav_window in tqdm(wav_windows):
        wav_window = wav_window.to(device)  # Shape: [1, window_size_samples]

        # Convert to Mel-spectrogram
        processed_mel = get_mel_spectrogram(wav_window, bigvgan_model.h).to(device)

        # Restore audio
        with torch.no_grad():
            with torch.autocast(device):
                restored_mel = model.voice_restore.sample(processed_mel.transpose(1, 2), steps=steps, cfg_strength=cfg_strength)
                restored_mel = restored_mel.squeeze(0).transpose(0, 1)

        # Convert restored mel-spectrogram to waveform
        with torch.no_grad():
            with torch.autocast(device):
                restored_wav = bigvgan_model(restored_mel.unsqueeze(0)).squeeze(0).float().cpu()  # Shape: [num_samples]
        
        # Debug: Print shapes
        # print(f"restored_wav.shape: {restored_wav.shape}")
        
        restored_wav_windows.append(restored_wav)
        del wav_window, processed_mel, restored_mel, restored_wav
        torch.cuda.empty_cache()

    restored_wav_windows = torch.stack(restored_wav_windows)  # Shape: [num_windows, num_samples]

    # Debug: Print shapes
    # print(f"restored_wav_windows.shape: {restored_wav_windows.shape}")

    # Reconstruct the full waveform from the processed windows
    restored_wav = reconstruct_waveform_from_windows(restored_wav_windows, window_size_samples, overlap)

    # Ensure the restored_wav has correct dimensions for saving
    if restored_wav.dim() == 1:
        restored_wav = restored_wav.unsqueeze(0)  # Shape: [1, num_samples]

    # Save the restored audio
    torchaudio.save(output_path, restored_wav, 24000)

    end_time = time.time()
    total_time = end_time - start_time
    peak_gpu_memory = measure_gpu_memory()
    gpu_memory_used = peak_gpu_memory - initial_gpu_memory

    print(f"Total inference time: {total_time:.2f} seconds")
    print(f"Peak GPU memory usage: {peak_gpu_memory:.2f} MB")
    print(f"GPU memory used: {gpu_memory_used:.2f} MB")

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Audio restoration using OptimizedAudioRestorationModel for long-form audio.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument('--input', type=str, required=True, help="Path to the input audio file")
    parser.add_argument('--output', type=str, required=True, help="Path to save the restored audio file")
    parser.add_argument('--steps', type=int, default=16, help="Number of sampling steps")
    parser.add_argument('--cfg_strength', type=float, default=0.5, help="CFG strength value")
    parser.add_argument('--window_size_sec', type=float, default=5.0, help="Window size in seconds for overlapping")
    parser.add_argument('--overlap', type=float, default=0.5, help="Overlap ratio for windowing")

    # Parse arguments
    args = parser.parse_args()

    # Load the optimized model
    optimized_model = load_model(args.checkpoint)

    if device == 'cuda':
        optimized_model.bfloat16()
    optimized_model.eval()
    optimized_model.to(device)

    # Use the model to restore audio
    restore_audio(
        optimized_model, 
        args.input, 
        args.output, 
        steps=args.steps, 
        cfg_strength=args.cfg_strength, 
        window_size_sec=args.window_size_sec, 
        overlap=args.overlap
    )
