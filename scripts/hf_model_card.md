---
license: other
license_name: dinov3-license-and-mit
tags:
- attribute-based-person-retrieval
- pedestrian-attribute-recognition
- upar
- dinov3
- clip
---

# UPAR Challenge 2027, Track 2 (attribute-based person retrieval): trained weights

Trained weights of our Track 2 leaderboard entry v13 (mADM 0.6817 on the hidden test set) to the UPAR Challenge 2027
(Real-World Surveillance workshop, WACV 2027). Code, training scripts and instructions: https://github.com/GITHUB_REPO .
The Codabench zip is rebuilt from these weights with `submission/build_v13.py`.

| File | What it is |
|---|---|
| `student.pt` | model 1: bf16 state dict of DINOv3 ViT-L/16 (176 x 96) with a query head, distilled for 20 epochs from DINOv3 ViT-L/16 and CLIP ViT-L/14 teachers (0.2 label loss + 0.8 distillation, half of the batches mixup / CutMix), seed 2 |
| `hp.pt` | model 2: bf16 state dict of DINOv3 ViT-H+/16 (224 x 112) with a query head, BCE + 0.5 group cross-entropy, EMA weights after 3 epochs, seed 1 |
| `m1.pt` | model 3: bf16 state dict of OpenAI CLIP ViT-L/14 (224 x 112) with a query head and the CLIP text score, same training as `hp.pt`, then WiSE-FT 0.7 |

All three models are trained on the train + val splits of PA100k, PETA and Market-1501.

Download:

```bash
hf download USER/REPO --local-dir weights      # huggingface_hub >= 1.0
```

The weights are fine-tuned from DINOv3 (DINOv3 License) and OpenAI CLIP (MIT License) and keep the terms of those
pretrained weights. The code is released under Apache 2.0.
