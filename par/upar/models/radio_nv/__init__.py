"""Patch generator of NVIDIA C-RADIO (CPE position embedding, per-teacher CLS tokens, registers), needed to rebuild
C-RADIO on top of a timm VisionTransformer without transformers.

vit_patch_generator.py and cls_token.py are unmodified files of https://huggingface.co/nvidia/C-RADIOv3-B
(NVIDIA Open Model License). They are not in git: radio._ensure_nv_files downloads them (pinned revision) on first use.
"""
