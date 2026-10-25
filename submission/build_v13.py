"""Build the Codabench code submission of the Track 2 leaderboard entry v13 (mADM 0.6817 on the hidden test set).

Released weights (rebuilds t2_stu_hpG_clipTGw70_rerank_gpu_v13.zip file for file, only the zip timestamps differ;
the two NVIDIA C-RADIO helper files of vendor/upar/models/radio_nv are included only if present, see README):
    python build_v13.py --student student.pt --hp hp.pt --m1 m1.pt --out t2_v13.zip
New training run (scripts/train_v13.sh):
    python build_v13.py --student student.pt --student-cfg student_cfg.json --hp hp.pt --hp-cfg hp_cfg.json \
        --m1 m1.pt --m1-cfg m1_cfg.json --out t2_v13.zip

The zip holds run.py, metadata.yaml, vendor/ (timm 1.0.30 and the upar model code), assets/config.json (three members,
logit average, no offset, no flip, kNN smoothing k = 50, alpha = 0.2 on the DINOv3 ViT-H+/16 features) and the bf16
weights assets/student.pt (DINOv3 ViT-L/16 student), assets/hp.pt (DINOv3 ViT-H+/16) and assets/m1.pt (CLIP ViT-L/14
with the text score, WiSE-FT 0.7).
"""
import argparse
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
V13 = HERE / "v13"
MEMBERS = ("student", "hp", "m1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True, help="bf16 weights of the DINOv3 ViT-L/16 student")
    ap.add_argument("--hp", required=True, help="bf16 weights of the DINOv3 ViT-H+/16 member")
    ap.add_argument("--m1", required=True, help="bf16 weights of the CLIP ViT-L/14 member (after WiSE-FT)")
    ap.add_argument("--out", required=True)
    for m in MEMBERS:
        ap.add_argument(f"--{m}-cfg", default=None, help="config written by par/export_member.py (new run only)")
    a = ap.parse_args()
    cfg_bytes = (V13 / "config.json").read_bytes()
    cfg = json.loads(cfg_bytes)
    assert [m["file"] for m in cfg["models"]] == [f"{m}.pt" for m in MEMBERS]
    assert cfg["offset_scale"] == 0.0 and cfg["tta"] is False
    assert cfg["rerank"] == {"member": "hp.pt", "k": 50, "alpha": 0.2}
    changed = False
    for i, m in enumerate(MEMBERS):
        path = getattr(a, f"{m}_cfg")
        if path:
            cfg["models"][i]["model"] = json.load(open(path))
            changed = True
    if changed:
        cfg_bytes = json.dumps(cfg, indent=1).encode()
    files = [HERE / "run.py", HERE / "metadata.yaml"] + sorted(p for p in (HERE / "vendor").rglob("*")
                                                                if p.is_file() and "__pycache__" not in p.parts)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(f, f.relative_to(HERE).as_posix())
        z.writestr("assets/config.json", cfg_bytes)
        for m in MEMBERS:
            z.write(getattr(a, m), f"assets/{m}.pt")
    print(a.out, len(files) + 4, "files", "(new config)" if changed else "(config of the submitted v13)")


if __name__ == "__main__":
    main()
