"""Track 2 (attribute-based person retrieval): query scoring and the official metrics.

A query is a 40-dim 0/1 attribute vector; a gallery image matches it if all 40 attributes are equal. Metrics follow
the Codabench evaluation page: AP, INP, Rank-k, and ADM (average degree of match), which credits near misses.
"""
import numpy as np


def score_l1(q, p):
    """Similarity = minus the expected number of mismatching attributes, sum_k |q_k - p_k| (the sample baseline).
    q (Q, K) in {0,1}, p (G, K) probabilities -> (Q, G); larger is better."""
    return -(p.sum(1)[None, :] + q @ (1.0 - 2.0 * p).T)


def score_loglik(q, p, eps=1e-4):
    """Similarity = log-probability that the image has exactly the query's attributes, assuming independent attributes:
    sum_k q_k log p_k + (1 - q_k) log(1 - p_k). Ranks by the chance of an exact match (what AP counts)."""
    p = np.clip(p, eps, 1 - eps)
    lp, lq = np.log(p), np.log1p(-p)
    return q @ (lp - lq).T + lq.sum(1)[None, :]


def retrieval_metrics(sim, q, g_attr, g_qid, q_ids, chunk=128):
    """sim (Q, G) larger = better; q (Q, K) query vectors; g_attr (G, K) gallery labels; g_qid (G,) query id of each
    gallery image; q_ids (Q,) id of each query row. Returns the mean of mADM, mAP, mINP, Rank-1/5/10 over queries."""
    Q, G = sim.shape
    K = q.shape[1]
    out = {"mADM": [], "mAP": [], "mINP": [], "Rank-1": [], "Rank-5": [], "Rank-10": []}
    gA = g_attr.astype(np.float32)
    for a in range(0, Q, chunk):
        b = min(Q, a + chunk)
        qq = q[a:b].astype(np.float32)
        eq = qq @ gA.T + (1 - qq) @ (1 - gA).T                  # (n, G) attributes equal to the query
        dom = eq / K
        mean, mx = dom.mean(1, keepdims=True), dom.max(1, keepdims=True)
        ndom = np.clip((dom - mean) / np.maximum(mx - mean, 1e-9), 0, None)
        order = np.argsort(-sim[a:b], axis=1, kind="stable")
        for i in range(b - a):
            o = order[i]
            match = g_qid[o] == q_ids[a + i]
            ranks = np.nonzero(match)[0] + 1                       # 1-based ranks of the correct matches
            if len(ranks) == 0:
                continue
            R = len(ranks)
            out["mAP"].append(float(np.mean(np.arange(1, R + 1) / ranks)))
            out["mINP"].append(R / ranks[-1])
            cs = np.cumsum(ndom[i][o])
            out["mADM"].append(float(np.mean(cs[ranks - 1] / ranks)))
            for k in (1, 5, 10):
                out[f"Rank-{k}"].append(float(ranks[0] <= k))
    return {k: float(np.mean(v)) for k, v in out.items()} | {"queries": len(out["mAP"])}
