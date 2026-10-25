"""bf16 weights + config of a finished train.py run, in the layout submission/build_v25.py takes.
python export_member.py RUN_DIR OUT_PT OUT_CFG_JSON      (run on the training machine; uses RUN_DIR/last.pt = EMA weights)
"""
import json, sys
import torch

ck = torch.load(f"{sys.argv[1]}/last.pt", map_location="cpu", weights_only=False)
torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in ck["model"].items()}, sys.argv[2])
json.dump(ck["model_cfg"], open(sys.argv[3], "w"), indent=1)
print(sys.argv[1], ck["model_cfg"]["backbone"], ck["model_cfg"]["size"], "epoch", ck.get("epoch"))
