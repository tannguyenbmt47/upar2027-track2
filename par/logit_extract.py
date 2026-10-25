"""Logits of a finished run on the whole val split, as the T1 package computes them (no flip, bf16 autocast).

python logit_extract.py ../runs/TR32_PETA_s1 out.npz [--wise 1.0 0.85 0.7]
  --wise a ...  WiSE-FT members: backbone weights (1 - a) * pretrained + a * fine-tuned, head kept (a = 1: the run)
 --wise-on both|h|c  dual model: which encoders are interpolated (default both)
  dual model (--backbone dual): also the three heads separately (h = H+ member, c = CLIP member, j = joint query)
Writes logits_w<a> (N_val, 40) [and h_w<a>, c_w<a>, j_w<a>], idx, holdout.
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from upar.data import UPARDataset, build_transforms
from upar.models import DualPARModel, build_model, cfg_from_checkpoint


@torch.no_grad()
def run_model(model, dl, N):
    dual = isinstance(model, DualPARModel)
    if dual:
        model.keep_parts = True
    out = {"logits": np.zeros((N, 40), np.float32)}
    n = 0
    for x, _, _, j, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x.cuda(non_blocking=True)).float()
        out["logits"][n:n + len(j)] = z.cpu().numpy()
        if dual:   # h = H+ head, c = CLIP head, x = third backbone's head (tri), j = joint query
            for k, v in model.parts.items():
                out.setdefault(k, np.zeros((N, 40), np.float32))[n:n + len(j)] = v.float().cpu().numpy()
        n += len(j)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("out")
    ap.add_argument("--wise", type=float, nargs="+", default=[1.0])
    ap.add_argument("--wise-on", default="both", choices=["both", "h", "c"],
                    help="dual model: interpolate both encoders, only H+ (h) or only CLIP (c)")
    a = ap.parse_args()
    ck = torch.load(f"{a.run}/last.pt", map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck)
    norm = {k: cfg[k] for k in ("mean", "std") if cfg.get(k)}
    ds = UPARDataset("val", build_transforms(tuple(cfg["size"]), False, **norm))
    dl = DataLoader(ds, 256, num_workers=16, pin_memory=True)
    pre = build_model(cfg, pretrained=True).state_dict() if any(w != 1.0 for w in a.wise) else None
    res = {"idx": np.arange(len(ds)), "holdout": str(ck["args"]["holdout"])}
    for w in a.wise:
        sd = ck["model"] if w == 1.0 else {
            k: ((1 - w) * pre[k].float() + w * v.float()).to(v.dtype)
            if ".backbone." in "." + k and k in pre and v.is_floating_point() and
            (a.wise_on == "both" or k.startswith(a.wise_on + ".")) else v for k, v in ck["model"].items()}
        model = build_model(cfg, pretrained=False).cuda().eval()
        model.load_state_dict(sd)
        for k, v in run_model(model, dl, len(ds)).items():
            res[f"{k}_w{w:.2f}"] = v
        print(f"{a.run} wise {w:.2f} done", flush=True)
        del model
    np.savez(a.out, **res)


if __name__ == "__main__":
    main()
