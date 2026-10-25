"""Fine-tune a backbone + head on UPAR with a pluggable loss.

Components are chosen by name (see upar/losses.py and upar/models/heads.py):

  python train.py --out ../runs/base                                   # linear head, BCE + DeepMAR (submission v3)
  python train.py --loss asl --loss-kw gamma_neg=4 clip=0.05 --out ../runs/asl
  python train.py --loss logit_adj --loss-kw tau=0.5 smooth=0.05 --out ../runs/la
  python train.py --head query --head-kw n_layers=1 --out ../runs/query
  python train.py --holdout PETA --epochs 5 ...                        # leave-one-dataset-out proxy
  python train.py --list                                               # show registered losses / heads
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import ModelEmaV3
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from upar.data import MEAN, STD, UPARDataset, attr_names, build_transforms, invalid_mask, load_split
from upar.losses import (LOSSES, NoiseAdaptation, build_loss, distill_bce, distill_missing, group_ce, listwise_ll,
                         soft_challenge, ONE_HOT_GROUPS, COLOUR_GROUPS, CONFUSABLE_COLOUR_GROUPS)
from upar.metrics import challenge_metrics, fmt
from upar.models import HEADS, build_model, cfg_from_checkpoint
from upar.registry import parse_kv
from upar.safety import guard


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("model")
    g.add_argument("--backbone", default="vit_base_patch16_dinov3.lvd1689m")
    g.add_argument("--size", type=int, nargs=2, default=[256, 128])
    g.add_argument("--head", default="linear", help=f"one of {HEADS.names()}")
    g.add_argument("--head-kw", nargs="*", default=[], metavar="K=V")
    g.add_argument("--ds-heads", action="store_true",
                   help="one classifier per source dataset (own labelling convention); inference averages them")
    g.add_argument("--keep-blocks", type=int, default=0, help="use only the first N backbone blocks (0 = all)")
    g.add_argument("--multi-layer", type=int, default=0, help="head reads a learned mix of the last N blocks' tokens")
    g.add_argument("--pos-init", default="resize", choices=["resize", "crop"],
                   help="absolute position embedding of a non-square input: resize the square grid (timm default) or "
                        "rescale isotropically to the height and centre-crop the width")
    g.add_argument("--text-init", default=None,
                   help="query_text head: file with CLIP projection + attribute text embeddings (make_clip_text.py)")
    g.add_argument("--dual-kw", nargs="*", default=[], metavar="K=V",
                   help="dual / tri models: mid=1 (mid-layer tokens), text_joint=1 (text-initialised joint queries), n_joint=2")
    g.add_argument("--branch-drop", type=float, default=0.0,
                   help="dual / tri models: per-batch probability that the joint query loses one backbone's tokens")
    g.add_argument("--cls-pool", action="store_true",
                   help="prepend the backbone's own pooled vector (SigLIP attention-pooling probe) as the CLS token")
    g.add_argument("--mixstyle-block", type=int, default=-1, help="MixStyle on the tokens after this block (-1 = off)")
    g.add_argument("--mixstyle-p", type=float, default=0.5)
    g.add_argument("--mixstyle-alpha", type=float, default=0.1)
    g = ap.add_argument_group("loss")
    g.add_argument("--loss", default="bce", help=f"one of {LOSSES.names()}")
    g.add_argument("--loss-kw", nargs="*", default=["smooth=0.05", "weighting=deepmar"], metavar="K=V",
                   help="shared: smooth, weighting (none|deepmar|sqrt|inverse), max_weight; plus loss-specific")
    g.add_argument("--noise-adapt", action="store_true",
                   help="per-dataset labelling-convention layer between the true attributes and the recorded labels")
    g.add_argument("--kd-missing", type=float, default=0.0,
                   help="with --teacher: weight of distillation on unlabelled cells only (instead of the usual KD)")
    g.add_argument("--aux-rec", default="none", choices=["none", "mae", "attr"],
                   help="training-only reconstruction decoder (dropped after training): masked patches or image from attributes")
    g.add_argument("--aux-rec-weight", type=float, default=0.2)
    g.add_argument("--moe-aux", type=float, default=0.01, help="weight of the MoE load-balancing loss")
    g.add_argument("--group-ce", type=float, default=0.0, help="weight of the one-of-K group cross-entropy term")
    g.add_argument("--gce-groups", default="all", choices=["all", "colours", "confusable"],
                   help="groups of the group cross-entropy: all one-hot groups, the two colour groups, or only the "
                        "colours models confuse most (black, blue, grey, white)")
    g.add_argument("--soft-metric", type=float, default=0.0,
                   help="weight of 1 - soft Challenge_Avg of the batch, on logits shifted by the decision rule")
    g.add_argument("--soft-metric-tau", type=float, default=0.42, help="tau of the decision rule inside --soft-metric")
    g.add_argument("--listwise", type=float, default=0.0, help="weight of the in-batch listwise log-likelihood loss")
    g.add_argument("--listwise-tau", type=float, default=1.0)
    g = ap.add_argument_group("distillation")
    g.add_argument("--teacher", default=None, help="teacher checkpoint (best.pt / last.pt of a bigger model)")
    g.add_argument("--kd-alpha", type=float, default=0.5, help="loss = (1-a) * label loss + a * distillation loss")
    g.add_argument("--kd-T", type=float, default=1.0, help="distillation temperature")
    g.add_argument("--kd-feat", type=float, default=0.0,
                   help="weight of patch-token feature distillation (per-teacher projector, z-scored teacher tokens)")
    g.add_argument("--kd-led", type=float, default=0.0,
                   help="weight of label-wise embedding structure distillation (L2D, ICCV 2023)")
    g.add_argument("--kd-rank", type=float, default=0.0,
                   help="weight of per-attribute listwise ranking distillation over the batch")
    g.add_argument("--kd-mix", type=float, default=0.0,
                   help="probability that a batch is mixed (mixup or cutmix, half each) and trained on the teachers' "
                        "outputs for the mixed images only (function matching, Beyer et al. 2022)")
    g.add_argument("--kd-agree", action="store_true",
                   help="weight each distillation cell by the teachers' agreement 1 - |p1 - p2| (needs 2+ teachers): "
                        "where they disagree the student leans on the label instead (after Beta-KD, CVPR 2026)")
    g = ap.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=10)
    g.add_argument("--bs", type=int, default=64)
    g.add_argument("--lr", type=float, default=5e-5, help="lr of the final backbone norm (the top block gets lr x layer decay)")
    g.add_argument("--head-lr-mult", type=float, default=10.0)
    g.add_argument("--wd", type=float, default=0.05)
    g.add_argument("--layer-decay", type=float, default=0.75)
    g.add_argument("--warmup", type=float, default=0.5, help="warmup epochs")
    g.add_argument("--ema", type=float, default=0.9995)
    g.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16"],
                   help="auto: bf16 on Ampere+ (A100, L4), fp16 + GradScaler on older GPUs (T4, V100)")
    g.add_argument("--grad-ckpt", action="store_true")
    g.add_argument("--lp-epochs", type=int, default=0,
                   help="LP-FT: train only the head for the first N epochs, then the whole network (Kumar et al. 2022)")
    g.add_argument("--freeze-blocks", type=int, default=0,
                   help="freeze patch embedding + the first N blocks (their layer-decayed lr is already ~0)")
    g.add_argument("--freeze-encoders", action="store_true",
                   help="dual models: freeze both encoders entirely (eval mode, no_grad); train only the query heads")
    g = ap.add_argument_group("data")
    g.add_argument("--train-splits", nargs="+", default=["train"])
    g.add_argument("--holdout", default=None, help="dataset left out of training; val is scored on it only")
    g.add_argument("--aug", default="base", choices=["base", "strong", "wb"], help="training augmentation strength")
    g.add_argument("--train-rows", default=None, help=".npy of train-split row indices to train on (cross-fitting)")
    g.add_argument("--noise-mask", default=None, help="npz of label cells to ignore, per split (not used by the submitted models)")
    g.add_argument("--ds-alpha", type=float, default=0.5, help="dataset sampling ~ N^alpha (0 = equal)")
    g.add_argument("--rare-rho", type=float, default=0.0,
                   help="fraction of each batch filled class-aware with rare attribute values (upar/sampling.py, 0 = off)")
    g.add_argument("--rare-thr", type=float, default=0.1, help="an attribute value is rare below this share")
    g.add_argument("--rare-power", type=float, default=0.5, help="rare keys drawn ~ mass^power (0 = uniform)")
    g.add_argument("--no-rare-comp", action="store_true",
                   help="do not compensate the sampler's prior shift in the loss (ablation)")
    g.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 2))
    g.add_argument("--prefetch", type=int, default=4, help="batches prefetched per worker (lower = less RAM)")
    g = ap.add_argument_group("run")
    g.add_argument("--out", default="../runs/base")
    g.add_argument("--no-best", action="store_true",
                   help="keep only last.pt (val logits of the last epoch): saves disk, val is not held out anyway")
    g.add_argument("--resume", action="store_true", help="continue from <out>/state.pt if it exists")
    g.add_argument("--seed", type=int, default=0, help="random seed (0 = not set)")
    g.add_argument("--data-seed", type=int, default=0,
                   help="reseed after building the model: same initialisation as --seed, different data order / "
                        "augmentation / drop-path (ingredients of a model soup must share the initialisation)")
    g.add_argument("--compile", action="store_true", help="torch.compile the training forward/backward (~+20%% img/s)")
    g.add_argument("--patience", type=int, default=0,
                   help="early stopping: stop after N epochs without a better val Challenge_Avg (0 = off)")
    g.add_argument("--max-iters", type=int, default=0, help="smoke test: stop each epoch after N iterations")
    g.add_argument("--list", action="store_true", help="print registered losses and heads and exit")
    args = ap.parse_args(argv)
    args.head_kw, args.loss_kw, args.dual_kw = parse_kv(args.head_kw), parse_kv(args.loss_kw), parse_kv(args.dual_kw)
    return args


def model_cfg(args):
    """What is needed to rebuild the network (stored in checkpoints, read by export_member.py / make_wise_member.py)."""
    mix = {"block": args.mixstyle_block, "p": args.mixstyle_p, "alpha": args.mixstyle_alpha} \
        if getattr(args, "mixstyle_block", -1) >= 0 else None
    return {"backbone": args.backbone, "size": list(args.size), "head": args.head, "head_kw": args.head_kw,
            "mean": getattr(args, "mean", None), "std": getattr(args, "std", None),
            "multi_layer": getattr(args, "multi_layer", 0), "mixstyle": mix, "keep_blocks": getattr(args, "keep_blocks", 0),
            "pos_init": getattr(args, "pos_init", "resize"), "cls_pool": getattr(args, "cls_pool", False),
            **({"branch_drop": args.branch_drop} if getattr(args, "branch_drop", 0) else {}),
            **({"dual_kw": args.dual_kw} if getattr(args, "dual_kw", None) else {})}


# ---------------------------------------------------------------- data
def build_data(args):
    size = tuple(args.size)

    def rows(split, keep):
        r = np.arange(len(load_split(split)))
        if args.holdout is not None:
            r = r[(load_split(split).ds.values[r] != args.holdout) == keep]
        if keep and split == "train" and args.train_rows:
            r = np.intersect1d(r, np.load(args.train_rows))
        return None if len(r) == len(load_split(split)) else r

    inv = invalid_mask(load_split("train"))
    norm = dict(mean=args.mean, std=args.std)
    noise = np.load(args.noise_mask) if args.noise_mask else {}
    train_sets = [UPARDataset(s, build_transforms(size, True, aug=args.aug, **norm), inv, rows=rows(s, True),
                              ignore=noise[s] if s in noise else None) for s in args.train_splits]
    weights = np.concatenate([d.sample_weights(args.ds_alpha) for d in train_sets])
    labels = np.concatenate([d.labels[d.idx] for d in train_sets])
    valid = np.concatenate([d.valid[d.idx] for d in train_sets])
    comp = torch.zeros(labels.shape[1])
    if args.rare_rho > 0:
        from upar.sampling import RareValueBatchSampler
        bsamp = RareValueBatchSampler(labels, valid, weights, args.bs, len(weights) // args.bs, args.rare_rho,
                                      args.rare_thr, args.rare_power, seed=args.seed + args.data_seed)
        print(bsamp.describe(attr_names()), flush=True)
        if not args.no_rare_comp:
            comp = bsamp.compensation()
        train_dl = DataLoader(ConcatDataset(train_sets), batch_sampler=bsamp, num_workers=args.workers,
                              pin_memory=True, persistent_workers=True, prefetch_factor=args.prefetch)
    else:
        sampler = WeightedRandomSampler(weights, sum(len(d) for d in train_sets), replacement=True)
        train_dl = DataLoader(ConcatDataset(train_sets), args.bs, sampler=sampler, num_workers=args.workers,
                              pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=args.prefetch)

    val_rows = rows("val", False)
    if args.max_iters:
        val_rows = (np.arange(len(load_split("val"))) if val_rows is None else val_rows)[::50]
    val_ds = UPARDataset("val", build_transforms(size, False, **norm), rows=val_rows)
    val_dl = DataLoader(val_ds, 128, num_workers=args.workers, pin_memory=True)

    prior = (labels * valid).sum(0) / valid.sum(0).clip(1)  # fraction of positives per attribute
    return train_dl, val_ds, val_dl, prior, comp


# ---------------------------------------------------------------- loops
@torch.no_grad()
def predict(model, loader, device="cuda", flip=True, dtype=torch.bfloat16):
    """Logits (hflip TTA averaged) and dataset row indices for every image of loader."""
    model.eval()
    logits, idx = [], []
    for x, _, _, j, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype):
            z = model(x).float()
            if flip:
                z = (z + model(torch.flip(x, dims=[3])).float()) / 2
        logits.append(z.cpu()); idx.append(j)
    return torch.cat(logits).numpy(), torch.cat(idx).numpy()


class CosineWithWarmup:
    def __init__(self, opt, total, warmup):
        self.opt, self.total, self.warmup = opt, total, warmup
        for g in opt.param_groups:
            g.setdefault("base_lr", g["lr"])

    def set(self, step):
        if step < self.warmup:
            s = step / max(1, self.warmup)
        else:
            s = 0.5 * (1 + math.cos(math.pi * (step - self.warmup) / max(1, self.total - self.warmup)))
        for g in self.opt.param_groups:
            g["lr"] = g["base_lr"] * s


class Teacher(nn.Module):
    """Frozen teacher(s) fed the student's batch. Each teacher gets the batch at its own input size and normalisation
    (the student's normalisation is undone first), and the logits are averaged: distilling an ensemble of different
    pretrainings (e.g. DINOv3 + CLIP) into one model that fits the CPU budget."""

    def __init__(self, models, cfgs, student_cfg):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.sizes = [tuple(c["size"]) for c in cfgs]
        stat = lambda c, k, d: torch.tensor(c.get(k) or d).view(1, 3, 1, 1)
        self.register_buffer("s_mean", stat(student_cfg, "mean", MEAN))
        self.register_buffer("s_std", stat(student_cfg, "std", STD))
        for i, c in enumerate(cfgs):
            self.register_buffer(f"mean{i}", stat(c, "mean", MEAN))
            self.register_buffer(f"std{i}", stat(c, "std", STD))
        self.same = [tuple(c["size"]) == tuple(student_cfg["size"]) and list(c.get("mean") or MEAN) == list(student_cfg.get("mean") or MEAN)
                     and list(c.get("std") or STD) == list(student_cfg.get("std") or STD) for c in cfgs]

    def forward(self, x):
        return self.forward_all(x).mean(0)

    def _inputs(self, x):
        raw = None
        for i in range(len(self.models)):
            if self.same[i]:
                yield x
                continue
            if raw is None:
                raw = x * self.s_std.to(x.dtype) + self.s_mean.to(x.dtype)
            xi = raw if self.sizes[i] == tuple(x.shape[-2:]) else F.interpolate(raw, size=self.sizes[i], mode="bilinear", align_corners=False)
            yield (xi - getattr(self, f"mean{i}").to(x.dtype)) / getattr(self, f"std{i}").to(x.dtype)

    def forward_rich(self, x):
        """Per teacher: logits (B, K), patch-token grid (B, D, h, w), attribute embeddings (B, K, D) or None."""
        out = []
        for i, (m, xi) in enumerate(zip(self.models, self._inputs(x))):
            logits, tok, q = m(xi, return_aux=True)
            ps = m.backbone.patch_embed.patch_size
            ps = ps[0] if isinstance(ps, (tuple, list)) else ps
            h, w = self.sizes[i][0] // ps, self.sizes[i][1] // ps
            pt = tok[:, -h * w:].float()
            out.append((logits.float(), pt.transpose(1, 2).reshape(pt.shape[0], -1, h, w), None if q is None else q.float()))
        return out

    def forward_all(self, x):
        """(n_teachers, B, K) logits."""
        raw = None
        out = []
        for i, m in enumerate(self.models):
            if self.same[i]:
                xi = x
            else:
                if raw is None:
                    raw = x * self.s_std.to(x.dtype) + self.s_mean.to(x.dtype)
                xi = raw if self.sizes[i] == tuple(x.shape[-2:]) else F.interpolate(raw, size=self.sizes[i], mode="bilinear", align_corners=False)
                xi = (xi - getattr(self, f"mean{i}").to(x.dtype)) / getattr(self, f"std{i}").to(x.dtype)
            out.append(m(xi).float())
        return torch.stack(out)


def load_teacher(paths, student_cfg):
    """Frozen teacher ensemble in eval mode from comma-separated checkpoints (sizes / normalisations may differ)."""
    models, cfgs = [], []
    for path in paths.split(","):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        tcfg = cfg_from_checkpoint(ck)
        teacher = build_model(tcfg, pretrained=False)
        teacher.load_state_dict(ck["model"])
        print(f"teacher {tcfg['backbone']} {tcfg['size']} from {path} (epoch {ck.get('epoch')}, "
              f"val {ck.get('val', {}).get('Challenge_Avg')})", flush=True)
        models.append(teacher)
        cfgs.append(tcfg)
    teacher = Teacher(models, cfgs, student_cfg)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher = teacher.cuda().eval()
    if torch.cuda.get_device_capability()[0] >= 8:
        teacher = teacher.to(torch.bfloat16)  # inference only: half the memory, next to the student on one GPU
    return teacher


def mix_batch(x):
    """Mixup (alpha 0.8) or cutmix (alpha 1.0) with a shuffled copy of the batch; the labels are not mixed: mixed
    batches are trained on the teachers' outputs for the mixed images."""
    perm = torch.randperm(x.shape[0], device=x.device)
    if np.random.rand() < 0.5:
        lam = float(np.random.beta(0.8, 0.8))
        return lam * x + (1 - lam) * x[perm]
    lam = float(np.random.beta(1.0, 1.0))
    H, W = x.shape[-2:]
    h, w = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y0, y1, x0, x1 = max(cy - h // 2, 0), min(cy + h // 2, H), max(cx - w // 2, 0), min(cx + w // 2, W)
    x = x.clone()
    x[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
    return x


def kd_loss(logits, t_all, args):
    """Distillation on every cell, optionally weighted by the agreement of the teachers."""
    t = t_all.mean(0)
    T = args.kd_T
    cell = F.binary_cross_entropy_with_logits(logits.float() / T, torch.sigmoid(t / T), reduction="none") * T * T
    if args.kd_agree and t_all.shape[0] > 1:
        p = torch.sigmoid(t_all)
        w = (1 - (p.max(0).values - p.min(0).values)).clamp(0, 1)
        return (cell * w).sum() / w.sum().clamp(min=1e-6)
    return cell.mean()


class KDExtra(nn.Module):
    """Training-only modules of the richer distillation losses (never saved with the student)."""

    def __init__(self, s_dim, t_dims):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(s_dim, d) for d in t_dims])


def feat_kd(kdx, s_tok, n_prefix, grid, rich):
    """PHI-S-style feature distillation: z-score each teacher's patch tokens per channel (balances teachers with very
    different activation scales), resize them to the student's patch grid, MSE against the projected student tokens."""
    h, w = grid
    sp = s_tok[:, -h * w:].float()
    loss = 0.0
    for proj, (_, tg, _) in zip(kdx.proj, rich):
        t = F.interpolate(tg, size=(h, w), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
        t = (t - t.mean((0, 1), keepdim=True)) / (t.std((0, 1), keepdim=True) + 1e-4)
        loss = loss + F.mse_loss(proj(sp), t)
    return loss / len(rich)


def led_kd(s_q, rich):
    """L2D label-wise embedding distillation: match the cosine-similarity structure of the attribute embeddings,
    between attributes of one image (A x A) and between images for one attribute (B x B); teachers averaged."""
    s = F.normalize(s_q.float(), dim=-1)
    ts = [F.normalize(q, dim=-1) for _, _, q in rich if q is not None]
    intra_t = sum(t @ t.transpose(1, 2) for t in ts) / len(ts)                       # (B, A, A)
    inter_t = sum(t.transpose(0, 1) @ t.permute(1, 2, 0) for t in ts) / len(ts)      # (A, B, B)
    return F.mse_loss(s @ s.transpose(1, 2), intra_t) + F.mse_loss(s.transpose(0, 1) @ s.permute(1, 2, 0), inter_t)


def rank_kd(logits, t_logits, tau=1.0):
    """Listwise ranking distillation per attribute: KL between the teacher's and the student's softmax over the batch."""
    pt = F.softmax(t_logits.float().T / tau, -1)
    return F.kl_div(F.log_softmax(logits.float().T / tau, -1), pt, reduction="batchmean")


RULE_SHIFT = None
SAMPLER_COMP = None


def train_one_epoch(ep, model, ema, loss_fn, opt, sched, scaler, loader, args, dtype, step, teacher=None, aux_dec=None,
                    adapt=None, kdx=None):
    from upar.models.heads import MoEFFN
    moe_modules = [m for m in model.modules() if isinstance(m, MoEFFN)]
    model.train(); t0 = time.time(); run = torch.zeros((), device="cuda")
    for it, (x, y, v, _, d) in enumerate(loader):
        if args.max_iters and it >= args.max_iters:
            break
        if it % 50 == 0:
            guard()
        sched.set(step)
        x, y, v = x.cuda(non_blocking=True), y.cuda(non_blocking=True), v.cuda(non_blocking=True)
        mixed = teacher is not None and args.kd_mix > 0 and np.random.rand() < args.kd_mix
        if mixed:
            x = mix_batch(x)
        with torch.autocast("cuda", dtype=dtype):
            ds_kw = {"ds": d.cuda(non_blocking=True)} if args.ds_heads else {}
            if aux_dec is not None or kdx is not None:
                logits, tokens, q_attr = model(x, return_aux=True, **ds_kw)
                if aux_dec is not None:
                    rec = aux_dec(q_attr if args.aux_rec == "attr" else tokens[:, model.n_prefix:], x)
            else:
                logits = model(x, **ds_kw)
        if adapt is not None:   # the labels are compared with what this dataset would record, not with the truth
            loss = loss_fn(adapt(logits, d.cuda(non_blocking=True)), y, v)
        else:
            loss = loss_fn(logits.float() + SAMPLER_COMP, y, v)
        if getattr(model, "parts", None) is not None:  # dual model: one loss per head (H+, CLIP, joint)
            if adapt is not None:   # each head's logits pass through the dataset's labelling convention
                dd = d.cuda(non_blocking=True)
                loss = sum(loss_fn(adapt(z, dd), y, v) for z in model.parts.values())
            else:
                loss = sum(loss_fn(z.float() + SAMPLER_COMP, y, v) for z in model.parts.values())
            model.parts = None
        if args.group_ce:
            gg = {"all": ONE_HOT_GROUPS, "colours": COLOUR_GROUPS, "confusable": CONFUSABLE_COLOUR_GROUPS}[args.gce_groups]
            loss = loss + args.group_ce * group_ce(logits, y, v, gg)
        if args.soft_metric:
            loss = loss + args.soft_metric * soft_challenge(logits, y, v, RULE_SHIFT)
        if args.listwise and not mixed:
            loss = loss + args.listwise * listwise_ll(logits, y, v, args.listwise_tau)
        if aux_dec is not None:
            loss = loss + args.aux_rec_weight * rec.float()
        aux = [m.last_aux for m in moe_modules if m.last_aux is not None]
        if aux:
            loss = loss + args.moe_aux * sum(aux)
        if teacher is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                if kdx is not None:
                    rich = teacher.forward_rich(x.to(next(teacher.parameters()).dtype))
                    t_all = torch.stack([r[0] for r in rich])
                else:
                    t_all = teacher.forward_all(x.to(next(teacher.parameters()).dtype)).float()
            t_logits = t_all.mean(0)
            if args.kd_missing:
                loss = loss + args.kd_missing * distill_missing(logits, t_logits, v)
            elif mixed:  # the labels do not describe a mixed image: teachers only
                loss = kd_loss(logits, t_all, args)
            elif args.kd_agree:
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd_loss(logits, t_all, args)
            else:
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * distill_bce(logits, t_logits, args.kd_T)
            if kdx is not None:
                grid = (args.size[0] // 16, args.size[1] // 16)
                with torch.autocast("cuda", dtype=dtype):
                    if args.kd_feat:
                        loss = loss + args.kd_feat * feat_kd(kdx, tokens, model.n_prefix, grid, rich)
                if args.kd_led:
                    loss = loss + args.kd_led * led_kd(q_attr, rich)
            if args.kd_rank:
                loss = loss + args.kd_rank * rank_kd(logits, t_logits)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + (list(aux_dec.parameters()) if aux_dec else [])
                                       + (list(kdx.parameters()) if kdx is not None else [])
                                       + (list(adapt.parameters()) if adapt is not None else []), 1.0)
        scaler.step(opt); scaler.update()
        ema.update(model, step=step); step += 1
        run += loss.detach()  # no .item() per step: it would make the CPU wait for the GPU
        if it % 200 == 0:
            print(f"ep {ep} it {it}/{len(loader)} loss {run.item() / (it + 1):.4f} lr {opt.param_groups[-1]['lr']:.2e} "
                  f"{(it + 1) * args.bs / (time.time() - t0):.0f} img/s", flush=True)
    return step


def evaluate(ep, model, val_ds, val_dl, dtype, t0):
    logits, idx = predict(model, val_dl, dtype=dtype)
    m, per = challenge_metrics(val_ds.labels[idx], 1 / (1 + np.exp(-logits)), val_ds.ds[idx])
    print(f"== ep {ep} ({time.time() - t0:.0f}s) VAL {fmt(m)}", flush=True)
    for d, pm in per.items():
        print(f"   {d:11s} {fmt(pm)}", flush=True)
    return m, logits, idx


# ---------------------------------------------------------------- main
def main():
    args = get_args()
    if args.list:
        print("losses:", LOSSES.names()); print("heads:", HEADS.names()); return
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.seed:
        import random
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.amp == "auto":
        args.amp = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    print(f"GPU {torch.cuda.get_device_name(0)}, amp {args.amp}, workers {args.workers}, "
          f"head {args.head} {args.head_kw}, loss {args.loss} {args.loss_kw}", flush=True)

    if args.ds_heads:  # classifiers of the datasets actually trained on are averaged at inference
        from upar.data import DOMAINS
        trained = [i for i, d in enumerate(DOMAINS) if d != args.holdout]
        args.head_kw = dict(args.head_kw, n_domains=len(DOMAINS), avg_domains=trained)
    model = build_model(model_cfg(args)).cuda()
    if args.data_seed:
        import random
        random.seed(args.data_seed); np.random.seed(args.data_seed); torch.manual_seed(args.data_seed)
    dcfg = model.data_config() if hasattr(model, "data_config") else \
        timm.data.resolve_data_config({}, model=model.backbone)  # input normalisation of the pretraining
    args.mean, args.std = list(dcfg["mean"]), list(dcfg["std"])
    print(f"input normalisation mean {args.mean} std {args.std}", flush=True)
    train_dl, val_ds, val_dl, prior, comp = build_data(args)
    for th in (model.text_heads() if hasattr(model, "text_heads") else [model.head]):
        if hasattr(th, "init_text"):  # query_text / inject_text: CLIP text anchors
            th.init_text(args.text_init, prior)
            print(f"text anchors from {args.text_init}", flush=True)
    global SAMPLER_COMP  # logit offset of the rare sampler's prior shift
    SAMPLER_COMP = comp.cuda()
    teacher = load_teacher(args.teacher, model_cfg(args)) if args.teacher else None
    if args.freeze_blocks:
        n_frozen = model.freeze_early(args.freeze_blocks)
        print(f"frozen patch embed + {args.freeze_blocks} blocks: {n_frozen / 1e6:.0f}M params", flush=True)
    if args.freeze_encoders:
        n_frozen = model.freeze_encoders()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"frozen encoders: {n_frozen / 1e6:.0f}M params, trainable {n_train / 1e6:.1f}M", flush=True)
    if args.grad_ckpt:
        (model if hasattr(model, "set_grad_checkpointing") else model.backbone).set_grad_checkpointing(True)
    loss_fn = build_loss(args.loss, prior, **args.loss_kw).cuda()
    global RULE_SHIFT  # decision rule tau * b_k with the training priors, used inside --soft-metric
    pi = torch.as_tensor(prior, dtype=torch.float32).clamp(1e-3, 1 - 1e-3)
    RULE_SHIFT = (args.soft_metric_tau * (torch.log((1 - pi) / pi) - (1 - 2 * pi))).cuda()
    ema = ModelEmaV3(model, decay=args.ema)
    if args.compile:  # in place: parameter names, EMA and checkpoints are unchanged
        model.compile()
    aux_dec = None
    if args.aux_rec != "none":  # trained with the head's lr, never saved: dropped after training
        from upar.models.aux_decoders import AUX_DECODERS
        aux_dec = AUX_DECODERS[args.aux_rec](model.backbone.num_features, (args.size[0] // 16, args.size[1] // 16)).cuda()
        print(f"aux decoder {args.aux_rec}: {sum(p.numel() for p in aux_dec.parameters()) / 1e6:.1f}M params, "
              f"weight {args.aux_rec_weight}", flush=True)
    adapt = None
    if args.noise_adapt:  # anchor = the training dataset with the most images (its convention defines "true")
        from upar.data import DOMAINS
        counts = {d: sum(int((ds.ds[ds.idx] == d).sum()) for ds in train_dl.dataset.datasets) for d in DOMAINS}
        anchor = DOMAINS.index(max(counts, key=counts.get))
        adapt = NoiseAdaptation(len(DOMAINS), len(attr_names()), anchor).cuda()
        print(f"noise adaptation, anchor {DOMAINS[anchor]}", flush=True)
    groups = model.param_groups(args.lr, args.wd, args.layer_decay, args.head_lr_mult)
    if adapt is not None:
        groups.append({"params": list(adapt.parameters()), "lr": args.lr * args.head_lr_mult, "weight_decay": 0.0})
    if aux_dec is not None:
        groups.append({"params": list(aux_dec.parameters()), "lr": args.lr * args.head_lr_mult, "weight_decay": args.wd})
    kdx = None
    if teacher is not None and (args.kd_feat or args.kd_led):
        t_dims = [m.backbone.num_features for m in teacher.models]
        kdx = KDExtra(model.backbone.num_features, t_dims).cuda()
        groups.append({"params": list(kdx.parameters()), "lr": args.lr * args.head_lr_mult, "weight_decay": args.wd})
        print(f"kd extras: feat {args.kd_feat} led {args.kd_led}, teacher dims {t_dims}", flush=True)
    opt = torch.optim.AdamW(groups, fused=True)
    sched = CosineWithWarmup(opt, args.epochs * len(train_dl), int(args.warmup * len(train_dl)))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")

    best, step, start, stale = -1.0, 0, 0, 0
    if args.resume and (out / "state.pt").exists():
        st = torch.load(out / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); ema.module.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
        best, step, start, stale = st["best"], st["step"], st["epoch"] + 1, st.get("stale", 0)
        print(f"resumed from epoch {st['epoch']} (best {best:.4f}, {stale} epoch(s) without improvement)", flush=True)

    trainable = [p for p in (model.backbone if hasattr(model, "backbone") else model).parameters() if p.requires_grad]
    for ep in range(start, args.epochs):
        t0 = time.time()
        if args.lp_epochs:  # LP-FT: backbone frozen while the freshly initialised head settles
            for p in trainable:
                p.requires_grad_(ep >= args.lp_epochs)
            print(f"epoch {ep}: {'head only (LP)' if ep < args.lp_epochs else 'full fine-tuning'}", flush=True)
        step = train_one_epoch(ep, model, ema, loss_fn, opt, sched, scaler, train_dl, args, dtype, step, teacher, aux_dec, adapt, kdx)
        if adapt is not None:  # learned conventions, for inspection
            ra, rb = adapt.rates()
            json.dump({"a": ra.tolist(), "b": rb.tolist()}, open(out / "noise_adapt.json", "w"))
        m, logits, idx = evaluate(ep, ema.module, val_ds, val_dl, dtype, t0)
        ck = {"model": ema.module.state_dict(), "model_cfg": model_cfg(args), "args": vars(args), "epoch": ep, "val": m}
        torch.save(ck, out / "last.pt")
        if args.no_best:
            np.savez(out / "val_logits.npz", logits=logits, idx=idx)
        if m["Challenge_Avg"] > best:
            best, stale = m["Challenge_Avg"], 0
            if not args.no_best:
                torch.save(ck, out / "best.pt")
                np.savez(out / "val_logits.npz", logits=logits, idx=idx)
        else:
            stale += 1
        if args.resume:  # full state so an interrupted Colab session can continue
            torch.save({"model": model.state_dict(), "ema": ema.module.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": ep, "step": step, "best": best, "stale": stale}, out / "state.pt.tmp")
            os.replace(out / "state.pt.tmp", out / "state.pt")
        if args.patience and stale >= args.patience:
            print(f"early stopping: no improvement for {stale} epochs (best {best:.4f})", flush=True)
            break
    print("best val Challenge_Avg", best)


if __name__ == "__main__":
    main()
