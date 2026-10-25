"""Prior offsets of the leaderboard entry v25.

offset_k = log((1 - p_k) / p_k) - (1 - 2 p_k), where p_k is the positive rate of attribute k over the train and val
splits of task 1 (131,076 images). run.py adds offset_scale * offset_k (offset_scale = 0.42) to the mean logit of the
two models before the sigmoid. The offsets depend only on the training labels, and the same 40 values are applied to
every test image.

python make_offsets.py ../UPAR-Challenge-2027/data/annotations/task1 --out offsets.json
python make_offsets.py ../UPAR-Challenge-2027/data/annotations/task1 --check ../submission/v25/config.json
"""
import argparse
import csv
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ann_dir", help="folder with train/gt.csv and val/gt.csv of task 1")
    ap.add_argument("--out", default=None, help="write {attribute_names, offset} as JSON")
    ap.add_argument("--check", default=None, help="compare with the offsets of a submission config.json")
    a = ap.parse_args()
    rows, names = [], None
    for split in ("train", "val"):
        with open(f"{a.ann_dir}/{split}/gt.csv", encoding="utf-8") as fh:
            r = csv.reader(fh)
            head = next(r)[1:]
            assert names in (None, head), "attribute order differs between splits"
            names = head
            rows += [[float(x) for x in row[1:]] for row in r if row]
    y = np.asarray(rows)
    p = y.mean(0)
    offset = np.log((1 - p) / p) - (1 - 2 * p)
    print(f"{len(y)} images, {len(names)} attributes, offsets {offset.min():.3f} .. {offset.max():.3f}")
    if a.out:
        json.dump({"attribute_names": names, "offset": offset.tolist()}, open(a.out, "w"), indent=1)
        print("wrote", a.out)
    if a.check:
        cfg = json.load(open(a.check))
        assert cfg["attribute_names"] == names, "attribute order differs from the config"
        print("max |difference| to", a.check, float(np.abs(np.asarray(cfg["offset"]) - offset).max()))


if __name__ == "__main__":
    main()
