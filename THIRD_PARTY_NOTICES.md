# Third-party code

| Path | Origin | License |
|---|---|---|
| `submission/vendor/timm/` | [timm](https://github.com/huggingface/pytorch-image-models) 1.0.30 by Ross Wightman, unmodified (vendored because the Codabench grader image has no timm) | Apache License 2.0 |
| `par/upar/models/radio_nv/`, `submission/vendor/upar/models/radio_nv/` | `__init__.py` only; the two NVIDIA files of [nvidia/C-RADIOv3-B](https://huggingface.co/nvidia/C-RADIOv3-B) are downloaded on first use and are not part of this repository | NVIDIA license |

Pretrained weights downloaded by the code: DINOv3 ViT-H+/16 (`vit_huge_plus_patch16_dinov3.lvd1689m`) and ViT-L/16
(`vit_large_patch16_dinov3.lvd1689m`), DINOv3 License, and OpenAI CLIP ViT-L/14 (`vit_large_patch14_clip_224.openai`,
MIT License). The released fine-tuned weights keep the
terms of these pretrained weights.
