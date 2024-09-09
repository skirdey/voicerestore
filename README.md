# VoiceRestore: Universal Audio Quality Restoration

VoiceRestore is a state-of-the-art audio restoration model that leverages flow-matching transformers to address a wide range of audio degradations. This repository contains the pretrained model and code necessary to run VoiceRestore on your own degraded audio files.

Demo of audio restorations: [VoiceRestore](https://sparkling-rabanadas-3082be.netlify.app/)

Credits: This repository is based on the [E2-TTS implementation by Lucidrains](https://github.com/lucidrains/e2-tts-pytorch)


Careful, heavily distorded audio. Adjust your volume before playing.


<audio controls>
  <source src="./audio/lq_heavy-distort-wall.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

## Key Features

- **Universal Restoration**: Handles multiple types of degradation including reverberation, noise, compression artifacts, and low sampling rates.
- **High Performance**: Achieves state-of-the-art results across various audio restoration benchmarks.
- **Easy to Use**: Simple interface for processing degraded audio files.
- **Pretrained Model**: Includes a 301 million parameter transformer model with pre-trained weights.

## Getting Started

 ```bash
git clone https://github.com/NVIDIA/BigVGAN.git
pip install torch torchaudio jaxtyping einops x-transformers torchdiffeq gateloop-transformer
```

- Download pre-trained checkpoint and place it into checkpoints folder. (TODO: will be released soon)

- Quick test: run below script should run on the test_input.wav file in the repository. The restored audio will be saved as test_output.wav.
```bash
python audio_restoration_model.py
```

- To process your audio files, use `model.forward(audio, steps=32, cfg_strength=1.0)` which will return audio tensor created by applying BigVGAN on the mel spectrogram output of the model. 


---
### Degraded Input: 

![Degraded Input](./imgs/degraded.png "Degraded Input")

### Restored (steps=32, cfg=1.0):

![Restored](./imgs/restored.png "Restored")


### Ground Truth:

![Ground Truth](./imgs/ground_truth.png "Ground Truth")



## Citation

If you use VoiceRestore in your research or projects, please cite our paper:

```
@article{kirdey2024voicerestore,
  title={VoiceRestore - Flow-Matching Transformers for Universal Audio Quality Restoration},
  author={Kirdey, Stanislav},
  journal={arXiv preprint arXiv:2024.XXXXX},
  year={2024}
}
```



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
