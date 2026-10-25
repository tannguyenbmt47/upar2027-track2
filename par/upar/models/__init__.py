"""Backbone (timm) + attribute head from the HEADS registry.

    model = build_model({"backbone": "vit_base_patch16_dinov3.lvd1689m", "size": [256, 128],
                         "head": "query", "head_kw": {"n_layers": 1}})

Only torch and timm are imported, so this subpackage is vendored into the submission as is.
"""
import contextlib

import timm
import torch
import torch.nn as nn

from .heads import HEADS
from .radio import create_radio, is_radio

DEFAULT_BACKBONE = "vit_base_patch16_dinov3.lvd1689m"


def _keep_input(module, inputs):
    module.kept_input = inputs[0]


def _keep_output(module, inputs, output):
    module.kept_output = output


class MixStyleHook:
    """Forward hook on a backbone block (train mode only): mixes the per-image channel mean / std of the patch tokens
    with those of another image of the batch (MixStyle, Zhou et al. 2021). The statistics carry camera / lighting /
    resolution "style"; mixing them across Market, PA100k and PETA makes the later blocks rely less on them."""

    def __init__(self, n_prefix, p=0.5, alpha=0.1, eps=1e-6):
        self.n_prefix, self.p, self.alpha, self.eps = n_prefix, p, alpha, eps

    def __call__(self, module, inputs, x):
        if not module.training:
            return x
        pre, tok = x[:, :self.n_prefix], x[:, self.n_prefix:]
        mu = tok.mean(1, keepdim=True).detach()
        sig = (tok.var(1, keepdim=True) + self.eps).sqrt().detach()
        B = x.shape[0]
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1)).to(x)
        apply = (torch.rand(B, 1, 1, device=x.device) < self.p).to(x)
        lam = 1 - apply * (1 - lam)  # lam = 1 (no mixing) for the samples that are not selected
        perm = torch.randperm(B, device=x.device)
        mu_mix, sig_mix = lam * mu + (1 - lam) * mu[perm], lam * sig + (1 - lam) * sig[perm]
        return torch.cat([pre, (tok - mu) / sig * sig_mix + mu_mix], 1)


class PARModel(nn.Module):
    def __init__(self, backbone=DEFAULT_BACKBONE, num_attrs=40, img_size=(256, 128), pretrained=True,
                 head="linear", head_kw=None, drop_path=0.1, multi_layer=0, mixstyle=None, keep_blocks=0,
                 pos_init="resize", cls_pool=False):
        super().__init__()
        if is_radio(backbone):  # NVIDIA C-RADIO: timm ViT + vendored patch generator, weights from Hugging Face
            self.backbone = create_radio(backbone, tuple(img_size), drop_path, pretrained=pretrained)
        else:
            kw = dict(pretrained=pretrained, num_classes=0, drop_path_rate=drop_path)
            try:  # ViTs take the input size (position embedding), CNNs do not
                self.backbone = timm.create_model(backbone, img_size=tuple(img_size), **kw)
            except TypeError:
                self.backbone = timm.create_model(backbone, **kw)
            if pos_init == "crop" and pretrained:
                self._crop_pos_embed(backbone, kw)
        if keep_blocks and hasattr(self.backbone, "blocks"):  # truncated backbone: first N blocks, cheaper on CPU
            self.backbone.blocks = self.backbone.blocks[:keep_blocks]
        # CNN (ConvNeXt, ...): feature map (B, C, H, W) -> tokens [global mean ; H*W cells], one prefix token
        # CNNs and SAM's ViT (its neck returns a (B, 256, H, W) map) give feature maps, not tokens
        self.is_cnn = not hasattr(self.backbone, "blocks") or backbone.startswith("samvit")
        if self.is_cnn and (multi_layer > 1 or mixstyle):
            raise ValueError("multi_layer / mixstyle need a ViT backbone")
        self.n_prefix = 1 if self.is_cnn else getattr(self.backbone, "num_prefix_tokens", 1)
        # backbones pretrained to be read out by a pooling head (SigLIP: attention-pooling probe, no CLS token): the
        # pooled vector, computed by the pretrained probe, is prepended as the CLS token the query head expects
        self.cls_pool = bool(cls_pool) and getattr(self.backbone, "attn_pool", None) is not None
        if self.cls_pool:
            self.n_prefix = 1
        head_kw = dict(head_kw or {})
        if (head in ("conv", "query_text_region") or head_kw.get("spatial_prior")) and "grid" not in head_kw:  # patch grid
            ps = getattr(self.backbone, "patch_size", None) or getattr(getattr(self.backbone, "patch_embed", None),
                                                                     "patch_size", 16)
            ps = ps[0] if isinstance(ps, (tuple, list)) else ps
            head_kw["grid"] = (img_size[0] // ps, img_size[1] // ps)
        self.head = HEADS.build(head, self.backbone.num_features, num_attrs, **head_kw)
        self.dense_path = head == "query_text_region"
        if self.dense_path:  # MaskCLIP dense embeddings need the input of the last block (kept on the block: EMA-safe)
            self.backbone.blocks[-1].register_forward_pre_hook(_keep_input)
        self.inject_at = None
        if head in ("inject", "inject_joint", "inject_text"):  # attribute tokens join the last k blocks as extra prefix tokens (no RoPE)
            blocks = self.backbone.blocks
            self.inject_at = len(blocks) - self.head.k
            for blk in blocks[self.inject_at:]:  # RoPE backbones (DINOv3) skip the prefix tokens; CLIP has no RoPE
                if hasattr(blk.attn, "num_prefix_tokens"):
                    blk.attn.num_prefix_tokens = blk.attn.num_prefix_tokens + num_attrs
        # tokens of the last `multi_layer` blocks, mixed with learned softmax weights (mid-level colour / texture cues)
        self.multi_layer = multi_layer
        if multi_layer > 1:
            self.neck = nn.ParameterDict({"layer_w": nn.Parameter(torch.zeros(multi_layer))})
            for blk in self.backbone.blocks[-multi_layer:]:  # output kept on the block itself: safe for the EMA deepcopy
                blk.register_forward_hook(_keep_output)
        if mixstyle:
            blocks = self.backbone.blocks
            pg = getattr(self.backbone, "patch_generator", None)  # C-RADIO: CLS + registers inside the blocks
            n_pre = pg.num_skip if pg is not None else self.n_prefix
            blocks[mixstyle["block"]].register_forward_hook(
                MixStyleHook(n_pre, mixstyle.get("p", 0.5), mixstyle.get("alpha", 0.1)))

    def _crop_pos_embed(self, backbone, kw):
        """Absolute position embedding for a non-square input without squeezing the pretrained grid: the native square
        grid is resized isotropically to the input height, then centre-cropped to the input width (timm's default
        resizes the grid to (h, w), which halves the horizontal spacing of a 2:1 crop). Only the initial value changes:
        the trained table is stored in the checkpoint, so inference code is unaffected. RoPE-only backbones (DINOv3)
        have no table and are left alone."""
        bb = self.backbone
        if getattr(bb, "pos_embed", None) is None or not hasattr(bb, "patch_embed"):
            return
        import torch.nn.functional as F
        native = timm.create_model(backbone, **kw)
        gh, gw = bb.patch_embed.grid_size
        n0h, n0w = native.patch_embed.grid_size
        n_pre = 0 if getattr(bb, "no_embed_class", False) else bb.num_prefix_tokens
        grid = native.pos_embed.data[:, n_pre:].reshape(1, n0h, n0w, -1).permute(0, 3, 1, 2).float()
        if n0h != gh:  # isotropic rescale by the height ratio
            scale = gh / n0h
            grid = F.interpolate(grid, size=(gh, max(gw, round(n0w * scale))), mode="bicubic", align_corners=False, antialias=True)
        w0 = grid.shape[3]
        if w0 < gw:
            grid = F.interpolate(grid, size=(gh, gw), mode="bicubic", align_corners=False)
        else:
            c = (w0 - gw) // 2
            grid = grid[:, :, :, c:c + gw]
        patch = grid.permute(0, 2, 3, 1).reshape(1, gh * gw, -1).to(bb.pos_embed.dtype)
        with torch.no_grad():
            bb.pos_embed[:, n_pre:].copy_(patch)
        del native

    def _inject_tokens(self, x):
        bb = self.backbone
        x = bb.patch_embed(x)
        x = bb._pos_embed(x)
        x, rope = x if isinstance(x, tuple) else (x, None)  # Eva / DINOv3 return the RoPE table, VisionTransformer not
        x = bb.norm_pre(x)
        for i, blk in enumerate(bb.blocks):
            if i == self.inject_at:
                a = self.head.attr_tokens.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
                x = torch.cat([x[:, :self.n_prefix], a, x[:, self.n_prefix:]], 1)
            x = blk(x, rope=rope) if rope is not None else blk(x)
        return bb.norm(x)

    def tokens(self, x):
        t = self._tokens(x)
        if self.cls_pool:
            g = self.backbone.forward_head(t, pre_logits=True)
            t = torch.cat([g[:, None].to(t.dtype), t], 1)
        return t

    def _tokens(self, x):
        if self.inject_at is not None:
            return self._inject_tokens(x)
        if self.is_cnn:
            t = self.backbone.forward_features(x).flatten(2).transpose(1, 2)
            return torch.cat([t.mean(1, keepdim=True), t], 1)
        if self.multi_layer <= 1:
            return self.backbone.forward_features(x)
        last = self.backbone.forward_features(x)
        w = self.neck["layer_w"].softmax(0)
        blocks = self.backbone.blocks[-self.multi_layer:]
        mixed = sum(wi * blk.kept_output for wi, blk in zip(w, blocks))
        for blk in blocks:
            blk.kept_output = None
        mixed = self.backbone.norm(mixed) if hasattr(self.backbone, "norm") else mixed
        if mixed.shape[1] != last.shape[1]:  # C-RADIO drops registers after the blocks: keep its token layout
            mixed = torch.cat([last[:, :1], mixed[:, -(last.shape[1] - 1):]], 1)
        return mixed

    def _dense_tokens(self):
        """MaskCLIP dense patch embeddings: value path of the last attention block, no query-key mixing, no residual / MLP,
        then the final norm (patch tokens only; the head applies the CLIP projection)."""
        blk = self.backbone.blocks[-1]
        xin = blk.kept_input; blk.kept_input = None
        B, N, C = xin.shape
        v = blk.attn.qkv(blk.norm1(xin)).reshape(B, N, 3, C)[:, :, 2]
        return self.backbone.norm(blk.attn.proj(v))[:, self.n_prefix:]

    def forward(self, x, ds=None, return_aux=False):
        t = self.tokens(x)
        if self.dense_path:
            self.head.dense = self._dense_tokens()
        args = (t, self.n_prefix) if ds is None else (t, self.n_prefix, ds)
        if not return_aux:
            return self.head(*args)
        if getattr(self.head, "supports_q", False):  # training-only decoders read tokens / attribute embeddings
            logits, q = self.head(*args, return_q=True)
        else:
            logits, q = self.head(*args), None
        return logits, t, q

    def freeze_early(self, n_blocks):
        """Freeze everything before block n_blocks (patch embed, cls / register tokens, rope, first blocks)."""
        n = 0
        for name, p in self.backbone.named_parameters():
            if name.startswith("blocks."):
                if int(name.split(".")[1]) >= n_blocks:
                    continue
            elif name.startswith(("norm", "fc_norm", "head", "neck", "attn_pool")):  # SAM neck sits after the blocks
                continue
            p.requires_grad_(False)
            n += p.numel()
        return n

    def param_groups(self, lr, weight_decay, layer_decay, head_lr_mult=10.0):
        """Layer-wise lr decay: the final backbone norm gets lr, the top block lr x layer_decay, each block below x layer_decay
        more, head x head_lr_mult."""
        cnn_layer = {}
        if self.is_cnn:  # timm's fine-grained parameter grouping (stem, then stage / block order)
            from timm.models import group_parameters
            for gid, names in group_parameters(self.backbone, self.backbone.group_matcher(coarse=False)).items():
                cnn_layer.update({"backbone." + nm: gid for nm in names})
            n = max(cnn_layer.values()) + 1
        else:
            n = len(self.backbone.blocks) + 1
        groups = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(("head.", "neck.")):
                layer, scale = n + 1, head_lr_mult
            elif name in cnn_layer:
                layer = cnn_layer[name]
                scale = layer_decay ** (n - 1 - layer)
            elif name.startswith("backbone.blocks."):
                layer = int(name.split(".")[2]) + 1
                scale = layer_decay ** (n - layer)
            elif name.startswith(("backbone.norm", "backbone.fc_norm", "backbone.attn_pool")):
                layer, scale = n, 1.0
            else:  # patch embed, cls / register tokens, rope
                layer, scale = 0, layer_decay ** n
            no_wd = p.ndim == 1 or "token" in name or "pos_embed" in name or "queries" in name
            wd = 0.0 if no_wd else weight_decay
            g = groups.setdefault((layer, wd), {"params": [], "lr": lr * scale, "weight_decay": wd})
            g["params"].append(p)
        return list(groups.values())


class DualPARModel(nn.Module):
    """DINOv3 ViT-H+ and OpenAI CLIP-L trained together in one model (--backbone dual).

    Each backbone keeps its own head, the two members of the T1 ensemble (H+: query, CLIP: query_text). A third, joint
    query head reads the tokens of both: every attribute query cross-attends to [CLS_H, patches_H, CLS_C, patches_C]
    after a per-source linear map to a shared width plus a learned source embedding, so it can take colour from one
    backbone and shape from the other in the same image. The input is in [0, 1]; each branch applies its own
    pretraining normalisation. In train mode the three logit sets are kept in self.parts (one loss per head);
    forward() returns their combination: "mean3" (default), "mean2" (the two members only) or "joint".
    """
    H, C = "vit_huge_plus_patch16_dinov3.lvd1689m", "vit_large_patch14_clip_224.openai"

    def __init__(self, num_attrs=40, img_size=(224, 112), pretrained=True, head_kw=None, combine="mean3", joint_dim=1024,
                 c_head="query_text", x_backbone=None, branch_drop=0.0, mid=False, text_joint=False, n_joint=1,
                 c_name=None, c_cls_pool=False, h_name=None, loc_read=False, loc_text=False, c_size=None,
                 joint_head="query", rscg_codes=2, rscg_v=1):
        super().__init__()
        hk = dict(head_kw or {})
        # h_name: another first encoder instead of DINOv3 ViT-H+ (--dual-kw h_name=nvidia/C-RADIOv4-H, ...)
        self.h = PARModel(h_name or self.H, num_attrs, img_size, pretrained=pretrained, head="query", head_kw=hk)
        # c_name: another second encoder instead of CLIP-L (--dual-kw c_name=<timm name> c_head=query c_cls_pool=1, e.g.
        # SigLIP2, whose attention-pooled vector stands in for the CLS token)
        # c_size: CLIP reads a resized copy of the input (e.g. DINOv3 at 288x144, CLIP at 224x112); "224,112" or a list
        self.c_size = tuple(int(v) for v in (c_size.split(",") if isinstance(c_size, str) else c_size)) if c_size else None
        self.c = PARModel(c_name or self.C, num_attrs, self.c_size or img_size, pretrained=pretrained, head=c_head,
                          head_kw=hk, cls_pool=bool(c_cls_pool))
        self.ph = nn.Linear(self.h.backbone.num_features, joint_dim)
        self.pc = nn.Linear(self.c.backbone.num_features, joint_dim)
        # optional third backbone (--backbone tri:<timm name>, a ViT-B): own query head + its tokens in the joint query
        self.x = PARModel(x_backbone, num_attrs, img_size, pretrained=pretrained, head="query", head_kw=hk,
                          cls_pool=True) if x_backbone else None
        self.px = nn.Linear(self.x.backbone.num_features, joint_dim) if x_backbone else None
        self.src = nn.Parameter(torch.zeros(3 if x_backbone else 2, joint_dim))
        # loc_read: the joint queries find WHERE to look on the DINOv3 tokens only (keys) and read, at those places, the
        # DINOv3 token plus the CLIP token resampled onto the DINOv3 patch grid (values): DINOv3 localises, CLIP names
        self.loc_read = int(loc_read)
        # joint_head=rscg: region-slot colour-geometry head (heads.py) over the [CLS_H, patches_H, CLS_C, patches_C] tokens
        if joint_head == "rscg":
            assert not self.loc_read and not x_backbone and not mid, "rscg reads the plain two-encoder token set"
            gh, gc = self.h.backbone.patch_embed.grid_size, self.c.backbone.patch_embed.grid_size
            self.joint = HEADS.build("rscg", joint_dim, num_attrs, n_tok=2 + gh[0] * gh[1] + gc[0] * gc[1],
                                     n_codes=int(rscg_codes), **hk,
                                     **({"cat": True, "grids": (tuple(gh), tuple(gc)), "codes": (2, 1)} if int(rscg_v) == 2 else {}))
        else:
            self.joint = HEADS.build("query", joint_dim, num_attrs, **hk, **({"sep_values": True} if self.loc_read else {}))
        # loc_read = 2: a second joint head the other way round (CLIP localises on its own grid, both encoders are read)
        self.joint_c = HEADS.build("query", joint_dim, num_attrs, **hk, sep_values=True) if self.loc_read == 2 else None
        # loc_text: located zero-shot score added to the joint logits. Small per-attribute queries find the attribute on
        # the DINOv3 patches; at those places the CLIP dense (MaskCLIP) embeddings, in CLIP's image-text space, are pooled
        # and compared with the CLIP head's text direction of the attribute (needs the query_text CLIP head)
        self.loc_text = bool(loc_text)
        if self.loc_text:
            self.lt_q = nn.Parameter(torch.randn(num_attrs, 256) * 0.02)
            self.lt_k = nn.Sequential(nn.LayerNorm(self.h.backbone.num_features), nn.Linear(self.h.backbone.num_features, 256))
            self.lt_gain = nn.Parameter(torch.tensor(0.1))
            if not self.c.dense_path:
                self.c.backbone.blocks[-1].register_forward_pre_hook(_keep_input)
        # n_joint = 2: a second joint query head on the same tokens (its own random queries); averaged like the others
        self.joint2 = HEADS.build("query", joint_dim, num_attrs, **hk) if n_joint > 1 else None
        # mid: the joint queries also read each backbone's tokens after the block at 2/3 depth (mid-level colour /
        # texture cues), through their own LayerNorm + linear map and source embeddings
        self.mid = bool(mid)
        if self.mid:
            self.mid_idx = {}
            for name, b in (("h", self.h), ("c", self.c)):
                i = len(b.backbone.blocks) * 2 // 3 - 1
                b.backbone.blocks[i].register_forward_hook(_keep_output)   # kept on the block itself: EMA-safe
                self.mid_idx[name] = i
            self.pmh = nn.Sequential(nn.LayerNorm(self.h.backbone.num_features), nn.Linear(self.h.backbone.num_features, joint_dim))
            self.pmc = nn.Sequential(nn.LayerNorm(self.c.backbone.num_features), nn.Linear(self.c.backbone.num_features, joint_dim))
            self.src_mid = nn.Parameter(torch.zeros(2, joint_dim))
        self.text_joint = bool(text_joint)
        for name, bb in (("h", self.h.backbone), ("c", self.c.backbone)) + ((("x", self.x.backbone),) if x_backbone else ()):
            d = timm.data.resolve_data_config({}, model=bb)
            self.register_buffer("mean_" + name, torch.tensor(d["mean"]).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("std_" + name, torch.tensor(d["std"]).view(1, 3, 1, 1), persistent=False)
        self.combine, self.n_prefix, self.parts, self.keep_parts = combine, 1, None, False
        # branch dropout (training only): with probability branch_drop per source, the joint query does not see that
        # backbone's tokens for this batch (never all of them), so it cannot rely on one source alone
        self.branch_drop = branch_drop
        self.frozen_enc = False   # freeze_encoders(): encoders fixed, run in eval mode under no_grad

    def data_config(self):
        return {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}

    def text_heads(self):
        from types import SimpleNamespace
        return [self.c.head] + ([SimpleNamespace(init_text=self.init_joint_text)] if self.text_joint else [])

    @torch.no_grad()
    def init_joint_text(self, path, prior):
        """text_joint: the joint queries start at the CLIP text direction of their attribute, mapped into CLIP's token
        space through the frozen projection (proj @ (t_attr - t_person)), and the CLIP token map pc starts as the
        identity, so at step 0 each joint query attends to the CLIP tokens that match its attribute description."""
        d = torch.load(path, map_location="cpu")
        q = (d["pos"] - d["neutral"][None]).float() @ d["proj"].float().T          # (A, CLIP width)
        assert q.shape[1] == self.pc.weight.shape[0] == self.pc.weight.shape[1], "text_joint needs joint_dim = CLIP width"
        qs = self.joint.queries
        qs.copy_(nn.functional.normalize(q, dim=-1).to(qs) * qs.norm(dim=-1, keepdim=True).mean())
        self.pc.weight.copy_(torch.eye(self.pc.weight.shape[0])); self.pc.bias.zero_()

    def branches(self):
        return [self.h, self.c] + ([self.x] if self.x is not None else [])

    def set_grad_checkpointing(self, enable=True):
        for b in self.branches():
            b.backbone.set_grad_checkpointing(enable)

    def freeze_encoders(self):
        """Freeze every encoder parameter, final norms included (--freeze-encoders). The encoders then run in eval
        mode (no drop path) under no_grad; only the query heads, the joint projections and the source embeddings
        are trained."""
        n = 0
        for b in self.branches():
            for p in b.backbone.parameters():
                p.requires_grad_(False)
                n += p.numel()
        self.frozen_enc = True
        self.train(self.training)
        return n

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "frozen_enc", False):
            for b in self.branches():
                b.backbone.eval()
        return self

    def freeze_early(self, n_blocks):
        """n_blocks for H+ (32 blocks) and CLIP-L (24); the 12-block third backbone freezes half as many."""
        n = self.h.freeze_early(n_blocks) + self.c.freeze_early(n_blocks)
        return n + (self.x.freeze_early(n_blocks // 2) if self.x is not None else 0)

    def param_groups(self, lr, weight_decay, layer_decay, head_lr_mult=10.0):
        groups = [g for b in self.branches() for g in b.param_groups(lr, weight_decay, layer_decay, head_lr_mult)]
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if p.requires_grad and name.startswith(("ph.", "pc.", "px.", "src", "joint", "pmh.", "pmc.", "lt_")):
                (no_decay if p.ndim <= 1 or name.startswith("src") or "queries" in name else decay).append(p)
        return groups + [{"params": decay, "lr": lr * head_lr_mult, "weight_decay": weight_decay},
                         {"params": no_decay, "lr": lr * head_lr_mult, "weight_decay": 0.0}]

    def forward(self, x, ds=None, return_aux=False):
        with torch.no_grad() if self.frozen_enc else contextlib.nullcontext():
            th = self.h.tokens((x - self.mean_h) / self.std_h)
            xc = x if self.c_size is None else nn.functional.interpolate(x, size=self.c_size, mode="bilinear",
                                                                          align_corners=False, antialias=True)
            tc = self.c.tokens((xc - self.mean_c) / self.std_c)
        dense_c = self.c._dense_tokens() if self.loc_text and not self.c.dense_path else None
        if self.c.dense_path:   # query_text_region: MaskCLIP dense embeddings for the colour region anchors
            self.c.head.dense = self.c._dense_tokens()
        zh, zc = self.h.head(th, self.h.n_prefix), self.c.head(tc, self.c.n_prefix)
        drop_reg = lambda t, n: torch.cat([t[:, :1], t[:, n:]], 1)   # noqa: E731  (CLS + patches, no registers)
        kv = [[self.ph(drop_reg(th, self.h.n_prefix)) + self.src[0]],   # one token group per backbone
              [self.pc(drop_reg(tc, self.c.n_prefix)) + self.src[1]]]
        if self.mid:
            for g, (name, b, proj) in enumerate((("h", self.h, self.pmh), ("c", self.c, self.pmc))):
                blk = b.backbone.blocks[self.mid_idx[name]]
                kv[g].append(proj(drop_reg(blk.kept_output, b.n_prefix)) + self.src_mid[g])
                blk.kept_output = None
        parts = {"h": zh, "c": zc}
        if self.x is not None:
            with torch.no_grad() if self.frozen_enc else contextlib.nullcontext():
                tx = self.x.tokens((x - self.mean_x) / self.std_x)
            parts["x"] = self.x.head(tx, self.x.n_prefix)
            kv.append([self.px(drop_reg(tx, self.x.n_prefix)) + self.src[2]])
        if self.training and self.branch_drop > 0:   # a dropped backbone loses all of its token groups
            keep = [i for i in range(len(kv)) if torch.rand(()) >= self.branch_drop] or [int(torch.randint(len(kv), ()))]
            kv = [kv[i] for i in keep]
        members = list(parts.values())
        if self.loc_read:
            hk, ck = drop_reg(th, self.h.n_prefix), drop_reg(tc, self.c.n_prefix)
            gh, gc = tuple(self.h.backbone.patch_embed.grid_size), tuple(self.c.backbone.patch_embed.grid_size)
            B, _, Dc = ck.shape
            cp = ck[:, 1:].transpose(1, 2).reshape(B, Dc, *gc).float()
            cp = nn.functional.interpolate(cp, size=gh, mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
            cr = torch.cat([ck[:, :1], cp.to(ck.dtype)], 1)                 # CLIP tokens on the DINOv3 grid
            keys = self.ph(hk) + self.src[0]
            kv = keys + self.pc(cr) + self.src[1]                            # values: both encoders at each place
            parts["j"] = self.joint(keys, 1, values=kv)
            if self.joint_c is not None:                                    # CLIP localises, both encoders are read
                Dh = hk.shape[-1]
                hp = hk[:, 1:].transpose(1, 2).reshape(B, Dh, *gh).float()
                hp = nn.functional.interpolate(hp, size=gc, mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
                hr = torch.cat([hk[:, :1], hp.to(hk.dtype)], 1)            # DINOv3 tokens on the CLIP grid
                keys_c = self.pc(ck) + self.src[1]
                parts["jc"] = self.joint_c(keys_c, 1, values=keys_c + self.ph(hr) + self.src[0])
            if self.loc_text:
                head = self.c.head
                dense = dense_c if dense_c is not None else self.c.head.dense
                e = nn.functional.normalize(dense.float() @ head.proj.float(), dim=-1)          # (B, Nc, 768)
                E = e.shape[-1]
                e = nn.functional.interpolate(e.transpose(1, 2).reshape(B, E, *gc), size=gh, mode="bilinear",
                                              align_corners=False).flatten(2).transpose(1, 2)    # (B, Nh, 768)
                k = self.lt_k(hk[:, 1:]).float()                                                 # (B, Nh, 256)
                a = torch.softmax(self.lt_q.float() @ k.transpose(1, 2) / 16.0, dim=-1)         # (B, A, Nh)
                pooled = nn.functional.normalize(a @ e, dim=-1)                                  # (B, A, 768)
                w = (head.text_w0 + head.delta_scale * head.text_delta).float()                 # (A, 768)
                lt = head.scale * (pooled * w[None]).sum(-1)                                     # (B, A)
                parts["j"] = parts["j"] + (self.lt_gain * lt).to(parts["j"].dtype)
        else:
            kv = torch.cat([t for grp in kv for t in grp], 1)
            parts["j"] = self.joint(kv, 1)
        if self.joint2 is not None:
            parts["j2"] = self.joint2(kv, 1)
        if self.training or self.keep_parts:
            self.parts = parts
        z = {"mean3": sum(parts.values()) / len(parts), "mean2": sum(members) / len(members), "joint": parts["j"]}[self.combine]
        return (z, kv, None) if return_aux else z


def build_model(cfg, pretrained=True):
    """cfg: dict with backbone, size and optional head / head_kw / num_attrs (train args or submission config)."""
    bb = cfg.get("backbone") or ""
    if bb in ("dual", "dual_tr") or bb.startswith("tri:"):   # dual_tr: CLIP query_text_region; tri:<timm>: + 3rd backbone
        kw = {"c_head": "query_text_region" if bb == "dual_tr" else "query_text", **(cfg.get("dual_kw") or {})}
        return DualPARModel(cfg.get("num_attrs", 40), cfg.get("size", (224, 112)), pretrained=pretrained,
                            head_kw=cfg.get("head_kw") or {}, combine=cfg.get("combine", "mean3"),
                            x_backbone=bb[4:] if bb.startswith("tri:") else None,
                            branch_drop=cfg.get("branch_drop", 0.0), **kw)
    return PARModel(cfg.get("backbone", DEFAULT_BACKBONE), cfg.get("num_attrs", 40), cfg.get("size", (256, 128)),
                    pretrained=pretrained, head=cfg.get("head", "linear"), head_kw=cfg.get("head_kw") or {},
                    multi_layer=cfg.get("multi_layer", 0), mixstyle=cfg.get("mixstyle"), keep_blocks=cfg.get("keep_blocks", 0),
                    pos_init=cfg.get("pos_init", "resize"), cls_pool=cfg.get("cls_pool", False))


def cfg_from_checkpoint(ck):
    """Model config of a checkpoint; checkpoints before the registry only stored backbone + size (linear head)."""
    if "model_cfg" in ck:
        return ck["model_cfg"]
    a = ck["args"]
    return {"backbone": a["backbone"], "size": a["size"], "head": a.get("head", "linear"), "head_kw": a.get("head_kw", {})}


__all__ = ["PARModel", "build_model", "cfg_from_checkpoint", "HEADS"]
