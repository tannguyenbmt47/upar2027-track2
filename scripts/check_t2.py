"""Run a built Track 2 submission zip the way the Codabench ingestion program does, on one validation gallery.

python scripts/check_t2.py t2_v13.zip [--dataset PETA] [--n 0]

The zip is extracted to a temporary folder, its run.py is imported and rank_gallery() gets
{"gallery": [{"image_path"}], "queries", "attribute_names"} for the validation images of one dataset and the
validation queries of those images. Output: mADM and mAP of the returned distances. The released models were trained
on train + val, so this is a smoke test of the package (loading, speed, output format), not a held-out score.
"""
import argparse
import importlib.util
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "par"))
from eval_retrieval import load_task2  # noqa: E402
from upar.data import DATA, attr_names, load_split  # noqa: E402
from upar.retrieval import retrieval_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--dataset", default="PETA", help="PETA, PA100k or Market1501")
    ap.add_argument("--n", type=int, default=0, help="first N gallery images only (0 = all)")
    a = ap.parse_args()
    A = attr_names()
    va = load_split("val")
    ids, Q = load_task2("val")
    qid_of = dict(zip(ids.image, ids.qid))
    rows = np.where(va.ds.values == a.dataset)[0]
    if a.n:
        rows = rows[:a.n]
    gq = np.array([qid_of[x] for x in va.image.values[rows]])
    qids = np.unique(gq)
    gallery = [{"image_path": str(DATA / va.image.values[r])} for r in rows]
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(a.zip).extractall(tmp)
        spec = importlib.util.spec_from_file_location("run", Path(tmp) / "run.py")
        run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run)
        t0 = time.time()
        out = run.rank_gallery({"gallery": gallery, "queries": Q[qids].tolist(), "attribute_names": A})
        dt = time.time() - t0
    met = retrieval_metrics(-np.asarray(out["distances"], np.float64), Q[qids], va[A].values[rows].astype(np.float32),
                            gq, qids)
    print(f"{a.dataset}: {len(rows)} gallery images, {len(qids)} queries, {dt:.1f}s; "
          f"mADM {met['mADM']:.4f} mAP {met['mAP']:.4f}")


if __name__ == "__main__":
    main()
