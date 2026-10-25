"""UPAR challenge metrics: label-based mA / F1, instance-based Acc / Prec / Rec / F1.

Each dataset is scored on its own, then every metric is averaged over datasets.
Challenge_Avg = harmonic mean of the dataset-averaged mA and Instance_F1.
Attributes without any positive in a dataset's ground truth are treated as invalid
for that dataset and excluded (the official GT marks them invalid instead).
"""
import numpy as np

EPS = 1e-20
KEYS = ["Challenge_Avg", "mA", "Label_F1", "Instance_Acc", "Instance_Prec", "Instance_Rec", "Instance_F1"]


def dataset_metrics(gt, pred, valid=None):
    """gt, pred: (N, A) {0,1}; valid: (A,) bool mask of attributes scored."""
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if valid is None:
        valid = gt.any(0)
    g, p = gt[:, valid], pred[:, valid]
    tp = (g & p).sum(0); tn = (~g & ~p).sum(0)
    fp = (~g & p).sum(0); fn = (g & ~p).sum(0)
    ma = ((tp / (tp + fn + EPS) + tn / (tn + fp + EPS)) / 2).mean()
    prec_l = tp / (tp + fp + EPS); rec_l = tp / (tp + fn + EPS)
    label_f1 = (2 * prec_l * rec_l / (prec_l + rec_l + EPS)).mean()

    inter = (g & p).sum(1); union = (g | p).sum(1)
    i_acc = inter / (union + EPS)
    i_prec = inter / (p.sum(1) + EPS)
    i_rec = inter / (g.sum(1) + EPS)
    i_f1 = 2 * i_prec * i_rec / (i_prec + i_rec + EPS)
    return dict(mA=ma, Label_F1=label_f1, Instance_Acc=i_acc.mean(), Instance_Prec=i_prec.mean(),
                Instance_Rec=i_rec.mean(), Instance_F1=i_f1.mean())


def challenge_metrics(gt, prob, groups, thr=0.5):
    """gt (N,A), prob (N,A) in [0,1], groups (N,) dataset name per row.
    Returns (mean metrics dict, per-dataset dict)."""
    per = {}
    for ds in np.unique(groups):
        m = groups == ds
        per[ds] = dataset_metrics(gt[m], prob[m] > thr)
        per[ds]["Challenge_Avg"] = hmean(per[ds]["mA"], per[ds]["Instance_F1"])
    mean = {k: float(np.mean([per[d][k] for d in per])) for k in per[next(iter(per))] if k != "Challenge_Avg"}
    mean["Challenge_Avg"] = hmean(mean["mA"], mean["Instance_F1"])
    return mean, per


def hmean(a, b):
    return 2 * a * b / (a + b + EPS)


def fmt(m):
    return " ".join(f"{k}={m[k]:.4f}" for k in KEYS if k in m)
