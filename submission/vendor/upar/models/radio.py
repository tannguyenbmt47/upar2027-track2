"""NVIDIA C-RADIO backbone (Hugging Face nvidia/C-RADIOv3-*) presented like a timm ViT.

C-RADIO is a timm VisionTransformer whose patch embedding is replaced by a "patch generator" (CPE position
embedding, one CLS token per teacher + registers). We rebuild exactly that from timm and the two vendored NVIDIA
files in radio_nv/, so training and the submission run the same code and the submission needs no transformers.
Pretrained weights are taken from the Hugging Face model (training only). Two changes to the original:

- forward_features returns [mean of the summary CLS tokens ; patch tokens], so heads see one prefix token;
- the position embedding always takes its eval path. In train mode CPE samples a random crop of the position
  grid (a pretraining augmentation); a short fine-tune would then train on positions it never sees at test time.

Input is normalised by the dataset transform with C-RADIO's own mean / std (exposed as pretrained_cfg),
so the model's input conditioner is not used.
"""
import shutil
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn

OPENAI_CLIP_MEAN, OPENAI_CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

# architecture of each release (config.json "args" of the Hugging Face repo)
SPECS = {
    "nvidia/C-RADIOv3-B": dict(embed_dim=768, depth=12, num_heads=12, n_teachers=4, summary_idxs=[0, 1, 2],
                               cpe_max_size=2048, num_registers=4),
}


# the two NVIDIA files of radio_nv/ are not in git (NVIDIA license): fetched once from the Hugging Face repo
NV_REPO, NV_REVISION = "nvidia/C-RADIOv3-B", "44653a0482cf460bb4f12595fc3cc3dfecc403d1"
NV_FILES = ("vit_patch_generator.py", "cls_token.py")


def _ensure_nv_files():
    d = Path(__file__).parent / "radio_nv"
    for f in NV_FILES:
        if not (d / f).exists():
            from huggingface_hub import hf_hub_download
            shutil.copy(hf_hub_download(NV_REPO, f, revision=NV_REVISION), d / f)


def is_radio(name):
    return name.startswith("nvidia/")


def _forward_tokens(self, x):  # enable_cpe_support._forward_cpe
    x = self.patch_generator(x)
    if getattr(self, "grad_checkpointing", False) and not torch.jit.is_scripting():
        from timm.models import checkpoint_seq
        x = checkpoint_seq(self.blocks, x)
    else:
        x = self.blocks(x)
    return self.norm(x)


def _pos_eval_path(self, batch_size, input_dims):
    """ViTPatchGenerator._get_pos_embeddings, always on its eval path. Bound method, not a closure: the EMA copy
    (deepcopy) must use its own position embedding, not the one of the model being trained."""
    was_training, self.training = self.training, False
    try:
        return type(self)._get_pos_embeddings(self, batch_size, input_dims)
    finally:
        self.training = was_training


def _forward_features(self, x):
    y = self.forward_tokens(x)
    return torch.cat([y[:, self.summary_idxs].mean(1, keepdim=True), y[:, self.patch_generator.num_skip:]], 1)


def _rename_grandma(state_dict, prefix, *args):
    """NVIDIA's Hugging Face code names the LayerScale weight 'grandma' (HF renames 'gamma' on load); timm's is 'gamma'.
    Accept both, so checkpoints of runs built from the Hugging Face model also load."""
    for k in [k for k in state_dict if k.startswith(prefix) and k.endswith(".grandma")]:
        state_dict[k[:-len("grandma")] + "gamma"] = state_dict.pop(k)


def build_radio_vit(name, img_size=(256, 128), drop_path=0.1):
    """Architecture only (random weights); mirrors radio_model.create_model_from_args + _enable_cpe_for_timm_vit."""
    from timm.models.vision_transformer import VisionTransformer
    _ensure_nv_files()  # present in the submission (copied by make_submission), downloaded otherwise
    from .radio_nv.vit_patch_generator import ViTPatchGenerator

    s = SPECS[name]
    vit = VisionTransformer(img_size=518 * 16 // 14, patch_size=16, embed_dim=s["embed_dim"], depth=s["depth"],
                            num_heads=s["num_heads"], init_values=1e-5, reg_tokens=4, no_embed_class=True,
                            num_classes=0, drop_path_rate=drop_path)
    vit.norm = nn.Identity()  # model_norm = False
    pg = ViTPatchGenerator(patch_size=16, embed_dim=s["embed_dim"], input_dims=vit.patch_embed.img_size,
                           normalize_patches=not isinstance(vit.patch_embed.norm, nn.Identity),
                           cls_token=vit.cls_token is not None, max_input_dims=s["cpe_max_size"], pos_dropout=0.1,
                           num_cls_tokens=s["n_teachers"], register_multiple=0, num_registers=s["num_registers"])
    vit.patch_generator = pg
    vit.patch_embed = vit.cls_token = vit.pos_embed = vit.pos_drop = None
    vit.patch_size, vit.num_cls_tokens, vit.num_registers = 16, s["n_teachers"], pg.num_registers

    pg._get_pos_embeddings = MethodType(_pos_eval_path, pg)
    vit.register_buffer("summary_idxs", torch.tensor(s["summary_idxs"]))
    vit._register_load_state_dict_pre_hook(_rename_grandma)
    vit.forward_tokens = MethodType(_forward_tokens, vit)
    vit.forward_features = MethodType(_forward_features, vit)
    vit.num_prefix_tokens = 1
    vit.pretrained_cfg = {"mean": OPENAI_CLIP_MEAN, "std": OPENAI_CLIP_STD, "input_size": (3, *img_size),
                          "interpolation": "bicubic", "crop_pct": 1.0}
    return vit


def create_radio(name, img_size=(256, 128), drop_path=0.1, pretrained=True):
    vit = build_radio_vit(name, img_size, drop_path)
    if pretrained:
        from transformers import AutoModel  # training only

        radio = AutoModel.from_pretrained(name, trust_remote_code=True).radio_model
        cond = radio.input_conditioner
        assert torch.allclose(cond.norm_mean.flatten(), torch.tensor(OPENAI_CLIP_MEAN)), "input normalisation changed"
        assert torch.allclose(cond.norm_std.flatten(), torch.tensor(OPENAI_CLIP_STD)), "input normalisation changed"
        assert radio.summary_idxs.tolist() == SPECS[name]["summary_idxs"], "summary tokens changed"
        state = dict(radio.model.state_dict(), summary_idxs=vit.summary_idxs)
        vit.load_state_dict(state, strict=True)
    return vit
