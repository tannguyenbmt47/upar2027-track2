"""UPAR 2027 Track 1 / Track 2 submission (GPU grader): logit-average ensemble of several attribute models.

The ingestion program calls load_model() once, then predict_batch(samples) (Track 1) or rank_gallery(sample) (Track 2).
Every model in config["models"] has its own backbone, input size and normalisation; each image is decoded once and
resized per model. Logits are averaged over the models (and over the horizontal flip when config["tta"]), then the
label-shift offset of the training priors is added (offset * offset_scale, the tau rule of the report) and the
sigmoid is returned. timm and upar.models are vendored in ./vendor (the evaluation image ships neither).
Track 2 with config["rerank"]: before ranking, each gallery image's logits are smoothed with those of its k nearest
gallery neighbours in the appearance space of one member (mean patch token, cosine), z' = (1 - alpha) z + alpha *
mean_kNN(z): an automatic re-ranking step applied identically to every gallery image.
No time budget and no fallback: a run that cannot finish fails and the previous leaderboard entry stays.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

T_IMPORT = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageFile  # noqa: E402
from upar.models import build_model  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

_CFG = None
_MODELS = []          # (model, (h, w), mean, std)
_OFFSET = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_AMP = None
_TTA = True
_POOL = None
_S = {"n": 0, "t0": None, "bad": 0}
_PREFETCH = {}  # image path -> future of its decoded tensors: decoding runs ahead of the GPU


def log(msg):
    print(f"[run.py] {msg}", flush=True)


def _norm(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def load_model() -> None:
    global _CFG, _OFFSET, _AMP, _TTA, _POOL
    t0 = time.time()
    with open(os.path.join(HERE, "assets", "config.json"), encoding="utf-8") as fh:
        _CFG = json.load(fh)
    n_attrs = len(_CFG["attribute_names"])
    if _DEVICE == "cuda":
        _AMP = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for m in _CFG["models"]:
        mcfg = dict(m["model"], num_attrs=n_attrs)
        model = build_model(mcfg, pretrained=False)
        state = torch.load(os.path.join(HERE, "assets", m["file"]), map_location="cpu", weights_only=True)
        model.load_state_dict({k: v.float() for k, v in state.items()})
        model = model.eval().to(_DEVICE)
        mean, std = (mcfg.get("mean") or IMAGENET[0]), (mcfg.get("std") or IMAGENET[1])
        _MODELS.append((model, tuple(mcfg["size"]), torch.tensor(mean, device=_DEVICE).view(1, 3, 1, 1),
                        torch.tensor(std, device=_DEVICE).view(1, 3, 1, 1)))
        log(f"model {m['file']}: {mcfg['backbone']} {mcfg['size']} head={mcfg.get('head')}")
    _TTA = bool(_CFG.get("tta", True))
    _OFFSET = torch.tensor(_CFG["offset"], dtype=torch.float32, device=_DEVICE) * float(_CFG.get("offset_scale", 1.0))
    _POOL = ThreadPoolExecutor(max_workers=max(1, min(18, os.cpu_count() or 1)))
    with torch.no_grad():  # warm-up / smoke test of every model
        for model, (h, w), mean, std in _MODELS:
            with torch.autocast("cuda", dtype=_AMP, enabled=_AMP is not None):
                out = model(torch.zeros(2, 3, h, w, device=_DEVICE)).float()
            if not torch.isfinite(out).all():
                raise RuntimeError("non-finite smoke test output")
    gpu = torch.cuda.get_device_name(0) if _DEVICE == "cuda" else "-"
    log(f"device={_DEVICE} gpu={gpu} amp={_AMP} tta={_TTA} models={len(_MODELS)} "
        f"offset_scale={_CFG.get('offset_scale')} load={time.time() - t0:.1f}s")


def _load(path):
    """One decode, one uint8 tensor per model input size."""
    sizes = [s for _, s, _, _ in _MODELS]
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            arrs = [np.asarray(im.resize((w, h), Image.BILINEAR), dtype=np.uint8) for h, w in sizes]
    except Exception as e:  # one unreadable file must not fail the run
        _S["bad"] += 1
        if _S["bad"] <= 5:
            log(f"WARNING cannot read {path}: {e!r}")
        arrs = [np.full((h, w, 3), 128, np.uint8) for h, w in sizes]
    return [torch.from_numpy(a.copy()).permute(2, 0, 1) for a in arrs]


def _prefetch(paths):
    for p in paths:
        if p not in _PREFETCH:
            _PREFETCH[p] = _POOL.submit(_load, p)


def _prefetch_from_csv(sample):
    """Queue every test image listed in images.csv (next to the images folder), in file order."""
    try:
        rel, full = sample.get("image"), sample["image_path"]
        if not rel or not full.endswith(rel):
            return 0
        img_dir = full[: len(full) - len(rel)].rstrip("/")
        csv = os.path.join(os.path.dirname(img_dir), "images.csv")
        if not os.path.exists(csv):
            return 0
        with open(csv, encoding="utf-8") as fh:
            rels = [line.strip().split(",")[0] for line in fh if line.strip()]
        _prefetch([os.path.join(img_dir, r) for r in rels if not r.startswith("#")])
        return len(rels)
    except Exception as e:
        log(f"prefetch skipped: {e!r}")
        return 0


def _order(names):
    ours = [_norm(n) for n in _CFG["attribute_names"]]
    try:
        return [ours.index(_norm(n)) for n in names]
    except ValueError:
        log(f"WARNING attribute names differ, using training order: {names}")
        return list(range(len(ours)))


def _batch_logits(samples: list[dict], feat_member=None):
    """Mean member logits of a batch (before the offset, training attribute order). With feat_member (index into
    _MODELS), also that member's L2-normalised mean patch token from the same forward pass (Track 2 re-ranking)."""
    if _S["t0"] is None:
        _S["t0"] = time.time()
        if not _PREFETCH:
            log(f"prefetching {_prefetch_from_csv(samples[0])} images from images.csv")
    paths = [s["image_path"] for s in samples]
    if all(p in _PREFETCH for p in paths):
        per_img = [_PREFETCH.pop(p).result() for p in paths]
    else:
        per_img = list(_POOL.map(_load, paths))
    logit, feat = 0.0, None
    for j, (model, _, mean, std) in enumerate(_MODELS):
        x = torch.stack([p[j] for p in per_img]).to(_DEVICE, non_blocking=True).float().div_(255)
        x = (x - mean) / std
        with torch.autocast("cuda", dtype=_AMP, enabled=_AMP is not None):
            if j == feat_member:
                t = model.tokens(x)
                z = model.head(t, model.n_prefix).float()
                feat = torch.nn.functional.normalize(t[:, model.n_prefix:].float().mean(1), dim=1)
            else:
                z = model(x).float()
            if _TTA:
                z = (z + model(torch.flip(x, dims=[3])).float()) / 2
        logit = logit + z
    logit = torch.nan_to_num(logit / len(_MODELS), nan=0.0, posinf=20.0, neginf=-20.0)
    _S["n"] += len(samples)
    if _S["n"] % 5120 < len(samples):
        dt = time.time() - _S["t0"]
        log(f"{_S['n']} images, {time.time() - T_IMPORT:.0f}s since import, {dt / _S['n'] * 1000:.2f} ms/img, bad={_S['bad']}")
    return logit, feat


@torch.no_grad()
def predict_batch(samples: list[dict]) -> list[list[float]]:
    if not _MODELS:
        load_model()
    names = samples[0].get("attribute_names") or _CFG["attribute_names"]
    logit, _ = _batch_logits(samples)
    out = torch.sigmoid(logit + _OFFSET).cpu().numpy().astype(np.float64)
    return out[:, _order(names)].tolist()


def predict_image(sample: dict) -> list[float]:
    return predict_batch([sample])[0]


def _knn_smooth(z, feat, k, alpha, chunk=4096):
    """z' = (1 - alpha) z + alpha * mean of z over the k most similar other gallery images (cosine of feat)."""
    g = z.shape[0]
    k = min(k, g - 1)
    if k < 1:  # a one-image gallery has no neighbours
        return z
    f = feat.half() if feat.is_cuda else feat
    idx = torch.empty(g, k, dtype=torch.long, device=z.device)
    for a in range(0, g, chunk):
        b = min(g, a + chunk)
        s = (f[a:b] @ f.T).float()
        s[torch.arange(b - a, device=s.device), torch.arange(a, b, device=s.device)] = -float("inf")
        idx[a:b] = s.topk(k, dim=1).indices
    return (1 - alpha) * z + alpha * z[idx].mean(1)


@torch.no_grad()
def rank_gallery(sample: dict) -> dict:
    """Track 2 entry point: {"distances": (Q, G)} = -(sum_k q_k log p_k + (1 - q_k) log(1 - p_k)), smaller = better.
    Use a config with offset_scale 0 (raw ensemble probabilities). config["rerank"] = {"member": file, "k", "alpha"}
    smooths the gallery logits over appearance neighbours first (see the module docstring)."""
    if not _MODELS:
        load_model()
    gallery, names = sample["gallery"], sample["attribute_names"]
    queries = np.asarray(sample["queries"], dtype=np.float64)
    log(f"queries: {len(queries)}, gallery images: {len(gallery)}")
    rr = _CFG.get("rerank")
    member = [m["file"] for m in _CFG["models"]].index(rr["member"]) if rr else None
    _prefetch([g["image_path"] for g in gallery])
    zs, fs = [], []
    for i in range(0, len(gallery), 128):
        z, f = _batch_logits([dict(g, attribute_names=names) for g in gallery[i:i + 128]], member)
        zs.append(z)
        fs.append(f)
    z = torch.cat(zs)
    if rr:
        t0 = time.time()
        z = _knn_smooth(z, torch.cat(fs), int(rr["k"]), float(rr["alpha"]))
        log(f"re-ranking: kNN smoothing k={rr['k']} alpha={rr['alpha']} on {rr['member']} features, {time.time() - t0:.1f}s")
    p = torch.sigmoid(z + _OFFSET).cpu().numpy().astype(np.float64)[:, _order(names)]
    p = np.clip(np.nan_to_num(p, nan=0.5), 1e-4, 1 - 1e-4)
    lp, lq = np.log(p), np.log1p(-p)
    sim = queries @ (lp - lq).T + lq.sum(1)[None, :]
    sim = np.nan_to_num(sim, nan=-1e6, posinf=1e6, neginf=-1e6).astype(np.float32)
    log(f"distances {sim.shape}, {time.time() - T_IMPORT:.0f}s since import")
    return {"distances": -sim}
