"""Rare-value batch sampler with prior compensation.

A batch of B images is filled in two parts:
  (1 - rho) B slots   drawn from the usual dataset-balanced distribution w (dataset d ~ N_d^alpha)
  rho B slots         class-aware: pick a rare (attribute, value) key, then an image holding that value (by w)
A key is rare when its share among the annotated cells under w is below `thr` (e.g. "LowerBody purple = 1",
"Female = 0" never qualifies). Keys are picked with probability ~ (mass of the key)^power, so power = 0.5 boosts
tail keys without repeating the few images of the rarest keys every batch (power 0 = uniform over keys).

The sampler changes the label frequencies the model sees, so its logits would drift away from the calibration the
test-time decision rule was tuned on. `compensation()` returns the per-attribute logit offset
    delta_k = logit(pi_k under the new sampler) - logit(pi_k under w)
which the training loss adds to the logits (logit adjustment, Menon et al. 2021). The network then still learns
the posterior of the dataset-balanced distribution, and inference is unchanged.
"""
import numpy as np
import torch
from torch.utils.data import Sampler


class RareValueBatchSampler(Sampler):
    def __init__(self, labels, valid, base_w, batch_size, n_batches, rho=0.25, thr=0.1, power=0.5, seed=0):
        """labels, valid: (N, A) of the concatenated training rows; base_w: (N,) sampling weights of the baseline."""
        self.bs, self.n_batches, self.seed, self.epoch = batch_size, n_batches, seed, 0
        self.n_rare = int(round(rho * batch_size))
        w = np.asarray(base_w, np.float64); self.p_base = w / w.sum()
        y, v = labels > 0.5, valid > 0.5
        keys, members, mass = [], [], []
        for k in range(labels.shape[1]):
            ann = self.p_base[v[:, k]].sum()
            if ann <= 0:
                continue
            for val in (True, False):
                rows = np.where(v[:, k] & (y[:, k] == val))[0]
                share = self.p_base[rows].sum() / ann
                if len(rows) and share < thr:
                    keys.append((k, int(val))); members.append(rows); mass.append(self.p_base[rows].sum())
        self.keys, self.members = keys, members
        kp = np.asarray(mass) ** power; self.key_p = kp / kp.sum()
        self.within = [self.p_base[r] / self.p_base[r].sum() for r in members]
        # exact per-image distribution of one slot, for the compensation
        q = np.zeros_like(self.p_base)
        for pk, r, pw in zip(self.key_p, members, self.within):
            q[r] += pk * pw
        f = self.n_rare / batch_size
        self.p_mix = (1 - f) * self.p_base + f * q
        self.labels, self.valid = labels, valid

    def compensation(self):
        def prior(p):
            num = (p[:, None] * self.labels * self.valid).sum(0); den = (p[:, None] * self.valid).sum(0)
            return np.clip(num / np.maximum(den, 1e-12), 1e-4, 1 - 1e-4)
        a, b = prior(self.p_mix), prior(self.p_base)
        return torch.as_tensor(np.log(a / (1 - a)) - np.log(b / (1 - b)), dtype=torch.float32)

    def describe(self, names):
        d = self.compensation().numpy()
        top = np.argsort(-np.abs(d))[:6]
        return (f"rare sampler: {len(self.keys)} rare keys, {self.n_rare}/{self.bs} slots per batch; largest prior "
                f"shifts (logit): " + ", ".join(f"{names[k]} {d[k]:+.2f}" for k in top))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch)); self.epoch += 1
        n_nat = self.bs - self.n_rare
        nat = rng.choice(len(self.p_base), size=(self.n_batches, n_nat), p=self.p_base)
        kid = rng.choice(len(self.keys), size=(self.n_batches, self.n_rare), p=self.key_p)
        for b in range(self.n_batches):
            rare = [self.members[k][rng.choice(len(self.members[k]), p=self.within[k])] for k in kid[b]]
            batch = np.concatenate([nat[b], np.asarray(rare, dtype=np.int64)])
            rng.shuffle(batch)
            yield batch.tolist()
