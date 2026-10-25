"""Multi-label losses for PAR.

Every loss shares three options handled by BaseLoss:
  smooth     label smoothing: targets become smooth/2 and 1 - smooth/2
  weighting  per-attribute weights from the training prior p (fraction of positives)
               none     w+ = w- = 1
               deepmar  w+ = exp(1 - p),  w- = exp(p)             (mild, ratio <= e)
               sqrt     w+ = sqrt((1 - p) / p), w- = 1            (strong, capped at max_weight)
               inverse  w+ = (1 - p) / p,       w- = 1            (very strong, capped at max_weight)
  masking    cells whose label is not annotated in a dataset (valid = 0) contribute nothing

Subclasses only define elementwise(logits, targets) -> per-cell loss.

  bce        binary cross-entropy
  logit_adj  BCE on logits shifted by tau * log(p / (1 - p)) during training only. At test time the raw
             logit is used, which moves the decision towards balanced accuracy (the mA view).
  logit_adj_rule  same, shift tau * (log(p / (1 - p)) + (1 - 2p)): with DeepMAR, the decision of the v25 rule
  focal      focal loss, down-weights easy cells: (1 - p_t)^gamma * BCE
  asl        asymmetric loss (Ridnik et al. 2021): focusing gamma_pos / gamma_neg and probability
             shifting (clip) for negatives, designed for many-negative multi-label problems
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import Registry

LOSSES = Registry("loss")


class BaseLoss(nn.Module):
    def __init__(self, prior, smooth=0.0, weighting="none", max_weight=20.0):
        super().__init__()
        p = torch.as_tensor(prior, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
        if weighting == "none":
            w_pos, w_neg = torch.ones_like(p), torch.ones_like(p)
        elif weighting == "deepmar":
            w_pos, w_neg = torch.exp(1 - p), torch.exp(p)
        elif weighting == "sqrt":
            w_pos, w_neg = ((1 - p) / p).sqrt().clamp(max=max_weight), torch.ones_like(p)
        elif weighting == "inverse":
            w_pos, w_neg = ((1 - p) / p).clamp(max=max_weight), torch.ones_like(p)
        else:
            raise ValueError(f"unknown weighting '{weighting}'")
        self.register_buffer("prior", p)
        self.register_buffer("w_pos", w_pos)
        self.register_buffer("w_neg", w_neg)
        self.smooth = smooth

    def elementwise(self, logits, targets):
        raise NotImplementedError

    def forward(self, logits, labels, valid):
        """logits, labels, valid: (B, A). labels in {0, 1}; valid = 0 masks unannotated cells."""
        targets = labels * (1 - self.smooth) + self.smooth / 2
        w = torch.where(labels > 0.5, self.w_pos, self.w_neg) * valid
        return (self.elementwise(logits.float(), targets) * w).sum() / valid.sum().clamp(min=1)


@LOSSES.register("bce")
class BCELoss(BaseLoss):
    def elementwise(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")


@LOSSES.register("logit_adj")
class LogitAdjustedBCE(BaseLoss):
    def __init__(self, prior, tau=1.0, **kw):
        super().__init__(prior, **kw)
        self.tau = tau

    def elementwise(self, logits, targets):
        shift = self.tau * torch.log(self.prior / (1 - self.prior))
        return F.binary_cross_entropy_with_logits(logits + shift, targets, reduction="none")


@LOSSES.register("logit_adj_rule")
class LogitAdjustedRuleBCE(BaseLoss):
    """logit_adj whose test-time decision equals the label-shift rule of the v25 package, learned during training.

    With DeepMAR weights the weighted-BCE optimum is z = logit(posterior) + (1 - 2p). The v25 package added
    tau * (log((1 - p) / p) - (1 - 2p)) to such logits after training. Training on z + s with
    s = tau * (log(p / (1 - p)) + (1 - 2p)) gives the same decision z > 0 with the raw logit (no output offset);
    plain logit_adj (s = tau * log(p / (1 - p))) keeps the full DeepMAR shift (1 - 2p) on top. Use with weighting=deepmar.
    """
    def __init__(self, prior, tau=0.42, **kw):
        super().__init__(prior, **kw)
        self.tau = tau

    def elementwise(self, logits, targets):
        p = self.prior
        shift = self.tau * (torch.log(p / (1 - p)) + (1 - 2 * p))
        return F.binary_cross_entropy_with_logits(logits + shift, targets, reduction="none")


@LOSSES.register("focal")
class FocalLoss(BaseLoss):
    def __init__(self, prior, gamma=2.0, **kw):
        super().__init__(prior, **kw)
        self.gamma = gamma

    def elementwise(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        return (1 - p_t).pow(self.gamma) * bce


@LOSSES.register("asl")
class AsymmetricLoss(BaseLoss):
    def __init__(self, prior, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, **kw):
        super().__init__(prior, **kw)
        self.gamma_pos, self.gamma_neg, self.clip = gamma_pos, gamma_neg, clip

    def elementwise(self, logits, targets):
        p = torch.sigmoid(logits)
        p_neg = (p - self.clip).clamp(min=0) if self.clip > 0 else p  # probability shifting
        loss_pos = targets * torch.log(p.clamp(min=1e-8)) * (1 - p).pow(self.gamma_pos)
        loss_neg = (1 - targets) * torch.log((1 - p_neg).clamp(min=1e-8)) * p_neg.pow(self.gamma_neg)
        return -(loss_pos + loss_neg)


def build_loss(name, prior, **kw):
    return LOSSES.build(name, prior, **kw)


def distill_bce(student_logits, teacher_logits, T=1.0):
    """Knowledge distillation for multi-label outputs: BCE between the student's and the teacher's tempered
    sigmoid, over every cell (also cells without a label in a dataset: the teacher fills them in).
    Scaled by T^2 so the gradient size does not depend on the temperature (Hinton et al. 2015)."""
    target = torch.sigmoid(teacher_logits.float() / T)
    return F.binary_cross_entropy_with_logits(student_logits.float() / T, target) * T * T


# attribute groups with exactly one positive per image in (almost) every annotated sample of all three datasets
ONE_HOT_GROUPS = ([0, 1, 2], [4, 5, 6], [33, 34], list(range(8, 20)), list(range(21, 33)))


def group_ce(logits, labels, valid, groups=ONE_HOT_GROUPS):
    """Categorical cross-entropy over each one-of-K group (softmax of the group's logits), on the samples whose group
    is fully annotated with exactly one positive. The 40 independent sigmoids put probability mass on label vectors
    that never occur (two ages, no upper colour); this term adds the correct categorical likelihood for 32 of them."""
    total, n = logits.new_zeros(()), 0
    for idx in groups:
        y, m = labels[:, idx], valid[:, idx]
        ok = (m.min(1).values > 0) & (y.sum(1) == 1)
        if ok.any():
            total = total + F.cross_entropy(logits[ok][:, idx].float(), y[ok].argmax(1), reduction="sum")
            n += int(ok.sum())
    return total / max(n, 1)


# colours that the models confuse most on held-out datasets: black, blue, grey, white (top, then bottom)
CONFUSABLE_COLOUR_GROUPS = ([8, 9, 12, 17], [21, 22, 25, 30])
COLOUR_GROUPS = (list(range(8, 20)), list(range(21, 33)))


def soft_challenge(logits, labels, valid, shift):
    """1 - soft Challenge_Avg of the batch: the harmonic mean of a soft mA (balanced accuracy per attribute over the
    batch) and a soft instance F1 (per image), computed on sigmoid(logits + shift), where shift is the test-time
    decision rule (tau * b_k). Trains the decisions the metric scores instead of a per-cell likelihood."""
    p = torch.sigmoid(logits.float() + shift)
    m = valid.float(); y = labels.float()
    pos, neg = y * m, (1 - y) * m
    npos, nneg = pos.sum(0), neg.sum(0)
    ok = (npos > 0) & (nneg > 0)
    tpr = (p * pos).sum(0) / npos.clamp(min=1); tnr = ((1 - p) * neg).sum(0) / nneg.clamp(min=1)
    ma = (0.5 * (tpr + tnr))[ok].mean() if ok.any() else p.new_ones(())
    tp = (p * pos).sum(1); fp = (p * neg).sum(1); fn = ((1 - p) * pos).sum(1)
    f1 = (2 * tp / (2 * tp + fp + fn).clamp(min=1e-6)).mean()
    return 1 - 2 * ma * f1 / (ma + f1).clamp(min=1e-6)


def listwise_ll(logits, labels, valid, tau=1.0):
    """Contrastive loss in the retrieval score space. Each image's label vector is used as a query, every image of the
    batch is scored with the Track 2 log-likelihood s_ij = sum_k y_ik log p_jk + (1 - y_ik) log(1 - p_jk) over the
    query's annotated attributes, and the loss is minus the log of the softmax mass (over the batch) on the images whose
    label vector equals the query's. This is the listwise criterion that predicts held-out mADM best (Spearman 0.97);
    inference is unchanged."""
    z = logits.float()
    lp, lq = F.logsigmoid(z), F.logsigmoid(-z)                      # log p, log (1 - p), (B, K)
    m = valid.float(); y = labels.float() * m
    S = (y @ lp.T + ((1 - labels.float()) * m) @ lq.T) / tau       # (B queries, B images)
    same = ((labels.float()[:, None, :] != labels.float()[None, :, :]).float() * m[:, None, :]).sum(-1) == 0
    return -(torch.logsumexp(S.masked_fill(~same, float("-inf")), 1) - torch.logsumexp(S, 1)).mean()


class NoiseAdaptation(torch.nn.Module):
    """Dataset-specific labelling conventions (noise adaptation layer, Goldberger & Ben-Reuven 2017).

    The model predicts the probability p that the person truly has the attribute. Dataset d records a positive with
    q = (1 - b) p + a (1 - p): a = rate of positives written for absent attributes, b = rate of present attributes
    left at 0, one (a, b) per dataset and attribute. The anchor dataset keeps a = b = 0 so that p is identifiable.
    Training compares q with the labels; inference uses p, so the layer is never part of the submission."""

    def __init__(self, n_domains, num_attrs, anchor, init=-4.0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.full((n_domains, num_attrs), init))
        self.b = torch.nn.Parameter(torch.full((n_domains, num_attrs), init))
        mask = torch.ones(n_domains, 1); mask[anchor] = 0
        self.register_buffer("mask", mask)

    def forward(self, logits, ds):
        """logits (B, A) of the true attributes, ds (B,) dataset ids -> logits of the recorded labels."""
        p = torch.sigmoid(logits.float())
        a = torch.sigmoid(self.a[ds]) * self.mask[ds]
        b = torch.sigmoid(self.b[ds]) * self.mask[ds]
        q = ((1 - b) * p + a * (1 - p)).clamp(1e-6, 1 - 1e-6)
        return torch.log(q) - torch.log1p(-q)

    def rates(self):
        return torch.sigmoid(self.a) * self.mask, torch.sigmoid(self.b) * self.mask


def distill_missing(student_logits, teacher_logits, valid):
    """BCE towards the teacher's probabilities on the cells without a label (valid = 0) only."""
    miss = 1 - valid
    target = torch.sigmoid(teacher_logits.float())
    l = F.binary_cross_entropy_with_logits(student_logits.float(), target, reduction="none")
    return (l * miss).sum() / miss.sum().clamp(min=1)
