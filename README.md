# voicerestore
VoiceRestore: Flow-Matching Transformers for Universal Audio Quality Restoration

Demo: [VoiceRestore](https://sparkling-rabanadas-3082be.netlify.app/)

```bash
pip install torch torchaudio bigvgan e2_tts jaxtyping einops x-transformers torchdiffeq gateloop-transformer
```

---

Degraded Input:
![Degraded Input](./imgs/degraded.png "Degraded Input")

Restored (steps=32, cfg=1.0):
![Restored](./imgs/restored.png "Restored")

Ground Truth:
![Ground Truth](./imgs/ground_truth.png "Ground Truth")