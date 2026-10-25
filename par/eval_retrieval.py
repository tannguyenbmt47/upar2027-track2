"""Evaluate Track 2 retrieval from attribute logits saved by a Track 1 run, one gallery per dataset of the val split.

python eval_retrieval.py <run dir> [--logits val_logits.npz] [--offsets offsets.json] [--scale 0.6]
The test gallery is a single unseen domain, so each val dataset is scored as its own gallery with its own queries.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from upar.data import DATA, attr_names, load_split
from upar.retrieval import retrieval_metrics, score_l1, score_loglik

A = attr_names()


def load_task2(split="val"):
    d = DATA / "annotations" / "task2" / split
    ids = pd.read_csv(d / "ids.csv", header=None, names=["image", "qid"])
    queries = np.loadtxt(d / "queries.csv", delimiter=",", comments="#").astype(np.float32)
    return ids, queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--logits", default="val_logits.npz")
    ap.add_argument("--offsets", default=None, help="offsets.json (Track 1 calibration); scaled by --scale")
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--datasets", nargs="+", default=["PA100k", "PETA", "Market1501"])
    args = ap.parse_args()
    run = Path(args.run)
    z = np.load(run / args.logits)
    logits, idx = z["logits"], z["idx"]
    if args.offsets:
        logits = logits + args.scale * np.asarray(json.load(open(run / args.offsets))["offset"])[None]
    prob = 1 / (1 + np.exp(-logits))
    va = load_split("val")
    ids, queries = load_task2("val")
    qid_of = dict(zip(ids.image, ids.qid))
    images, ds, gt = va.image.values[idx], va.ds.values[idx], va[A].values[idx].astype(np.float32)
    gq = np.array([qid_of[p] for p in images])
    print(f"{run.name}  logits={args.logits}  offsets={args.offsets} x{args.scale if args.offsets else 0}")
    for d in args.datasets:
        m = ds == d
        if not m.any():
            continue
        q_ids = np.unique(gq[m])
        for name, fn in (("L1", score_l1), ("loglik", score_loglik)):
            r = retrieval_metrics(fn(queries[q_ids], prob[m]), queries[q_ids], gt[m], gq[m], q_ids)
            print(f"  {d:11s} {name:7s} gallery {m.sum():5d} queries {r['queries']:4d} | mADM {r['mADM']:.4f}  mAP {r['mAP']:.4f}  "
                  f"R1 {r['Rank-1']:.4f}  R10 {r['Rank-10']:.4f}  mINP {r['mINP']:.4f}")


if __name__ == "__main__":
    main()
