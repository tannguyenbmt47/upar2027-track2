"""UPAR data: annotation loading, encoded-image cache, dataset, transforms."""
import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms as T

DATA = Path(__file__).resolve().parents[2] / "UPAR-Challenge-2027" / "data"
CACHE = Path(__file__).resolve().parents[2] / "cache"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_split(split):
    df = pd.read_csv(DATA / f"annotations/task1/{split}/gt.csv")
    df.columns = ["image"] + list(df.columns[1:])
    df["ds"] = df.image.str.split("/").str[0]
    return df


def attr_names():
    return list(load_split("val").columns[1:41])


DOMAINS = ["Market1501", "PA100k", "PETA"]  # fixed ids for dataset-specific heads


def invalid_mask(df_train):
    """(n_ds, A) bool: attribute never positive in a dataset's train labels -> treated as unannotated."""
    attrs = attr_names()
    return {ds: (g[attrs].values.sum(0) == 0) for ds, g in df_train.groupby("ds")}


def image_cache(split):
    """All encoded image bytes of a split in one uint8 array + offsets (fork/COW friendly)."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{split}_bytes.npz"
    if not f.exists():
        df = load_split(split)
        blobs = [(DATA / p).read_bytes() for p in df.image]
        off = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
        np.savez(f, buf=np.frombuffer(b"".join(blobs), np.uint8), off=off)
    z = np.load(f)
    return z["buf"], z["off"]


class RandomLowRes:
    """Simulate low-resolution / far-distance crops: downscale then upscale."""

    def __init__(self, p=0.3, min_scale=0.3):
        self.p, self.min_scale = p, min_scale

    def __call__(self, img):
        if random.random() > self.p:
            return img
        s = random.uniform(self.min_scale, 0.8)
        w, h = img.size
        small = img.resize((max(8, int(w * s)), max(16, int(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class RandomBlur:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5))) if random.random() < self.p else img


class RandomJPEG:
    """Re-encode with a random JPEG quality: surveillance crops are compressed, often heavily."""

    def __init__(self, p=0.5, quality=(30, 90)):
        self.p, self.quality = p, quality

    def __call__(self, img):
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.quality))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


class RandomWhiteBalance:
    """Per-channel gain and gamma: camera colour cast / exposure without changing hue ordering much."""

    def __init__(self, p=0.3, gain=0.1, gamma=(0.8, 1.25)):
        self.p, self.gain, self.gamma = p, gain, gamma

    def __call__(self, x):  # tensor in [0, 1]
        if random.random() > self.p:
            return x
        g = 1 + (torch.rand(3, 1, 1) * 2 - 1) * self.gain
        return (x * g).clamp(0, 1) ** random.uniform(*self.gamma)


class RandomNoise:
    def __init__(self, p=0.2, sigma=(0.01, 0.04)):
        self.p, self.sigma = p, sigma

    def __call__(self, x):
        if random.random() > self.p:
            return x
        return (x + torch.randn_like(x) * random.uniform(*self.sigma)).clamp(0, 1)


def build_transforms(size, train, mean=MEAN, std=STD, aug="base"):
    """mean / std must match the backbone's pretraining (ImageNet for DINOv3, 0.5 for PE-Core / SigLIP)."""
    h, w = size
    if not train:
        return T.Compose([T.Resize((h, w)), T.ToTensor(), T.Normalize(mean, std)])
    if aug == "strong":
        # camera / capture shift: resolution, compression, framing, viewpoint, occlusion, colour cast. No hue change.
        return T.Compose([
            RandomLowRes(p=0.5, min_scale=0.2),
            RandomJPEG(p=0.5, quality=(30, 90)),
            T.RandomResizedCrop((h, w), scale=(0.8, 1.0), ratio=(w / h * 0.8, w / h * 1.25), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.RandomAffine(degrees=8, translate=(0.05, 0.05))], p=0.3),
            T.RandomPerspective(distortion_scale=0.15, p=0.2),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
            RandomBlur(p=0.2),
            T.ToTensor(),
            RandomWhiteBalance(p=0.3),
            RandomNoise(p=0.2),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.5, scale=(0.02, 0.25), value="random"),
        ])
    wb = [RandomWhiteBalance(p=0.5, gain=0.15, gamma=(0.9, 1.1))] if aug == "wb" else []
    return T.Compose([
        RandomLowRes(),
        T.Resize((h, w)),
        T.RandomHorizontalFlip(),
        T.Pad(10, padding_mode="edge"),
        T.RandomCrop((h, w)),
        # no hue jitter: colour is a label
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        RandomBlur(),
        T.ToTensor(),
        *wb,  # "wb": base + per-channel gains (camera / illuminant colour cast, von Kries model)
        T.Normalize(mean, std),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15), value="random"),
    ])


class UPARDataset(Dataset):
    def __init__(self, split, transform, invalid=None, rows=None, ignore=None):
        df = load_split(split)
        self.buf, self.off = image_cache(split)
        self.idx = np.arange(len(df)) if rows is None else np.asarray(rows)
        attrs = attr_names()
        self.labels = df[attrs].values.astype(np.float32)
        self.ds = df.ds.values
        self.ds_names = sorted(set(self.ds))
        valid = np.ones_like(self.labels)
        if invalid:
            for ds, m in invalid.items():
                valid[self.ds == ds] = ~m
        if ignore is not None:  # (len(split), A) bool: cells flagged as label noise are treated as unannotated
            valid[ignore] = 0
        self.valid = valid.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        img = Image.open(io.BytesIO(self.buf[self.off[j]:self.off[j + 1]].tobytes())).convert("RGB")
        return self.transform(img), torch.from_numpy(self.labels[j]), torch.from_numpy(self.valid[j]), j, \
            DOMAINS.index(self.ds[j])

    def sample_weights(self, alpha=0.5):
        """Per-sample weights so dataset d is drawn ~ N_d^alpha (alpha=0: equal per dataset)."""
        ds = self.ds[self.idx]
        names, counts = np.unique(ds, return_counts=True)
        per = {n: c ** alpha / c for n, c in zip(names, counts)}
        return np.array([per[d] for d in ds])
