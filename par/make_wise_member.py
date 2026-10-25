"""WiSE-FT member: backbone weights (1 - a) * pretrained + a * fine-tuned of a finished run (head and text anchors kept),
written as a new run directory (last.pt + args.json) that export_member.py takes like any other run.

python make_wise_member.py ../runs/F_clipT 0.7 ../runs/F_clipT_w70
"""
import json
import shutil
import sys
from pathlib import Path

import torch

from upar.models import build_model, cfg_from_checkpoint

run, a, out = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3])
ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
pre = build_model(cfg_from_checkpoint(ck), pretrained=True).state_dict()
n = 0
for k, v in ck["model"].items():
    if k.startswith("backbone.") and k in pre and v.is_floating_point():
        ck["model"][k] = ((1 - a) * pre[k].float() + a * v.float()).to(v.dtype); n += 1
ck["wise_alpha"] = a
out.mkdir(parents=True, exist_ok=True)
torch.save(ck, out / "last.pt")
shutil.copy(run / "args.json", out / "args.json")
print(f"{out}: {n} backbone tensors interpolated with alpha {a}")
