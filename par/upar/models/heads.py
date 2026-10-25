"""Attribute heads. A head maps backbone tokens (B, T, D) to logits (B, A).

tokens[:, 0] is the CLS token, tokens[:, :n_prefix] are CLS + register tokens,
tokens[:, n_prefix:] are the patch tokens.
"""
import torch
import torch.nn as nn

from ..registry import Registry

HEADS = Registry("head")


@HEADS.register("linear")
class LinearHead(nn.Sequential):
    """[CLS ; mean(patches)] -> LayerNorm -> Dropout -> Linear. One shared feature for all attributes.

    Subclasses nn.Sequential so parameter names stay head.0.* / head.2.* (checkpoints of submission v3).
    """

    def __init__(self, dim, num_attrs, drop=0.1):
        super().__init__(nn.LayerNorm(2 * dim), nn.Dropout(drop), nn.Linear(2 * dim, num_attrs))

    def forward(self, tokens, n_prefix):
        feat = torch.cat([tokens[:, 0], tokens[:, n_prefix:].mean(1)], 1)
        return super().forward(feat)


class SwiGLU(nn.Module):
    """Gated MLP (Shazeer 2020; used in LLaMA and DINOv3): W3 (SiLU(W1 x) * W2 x). The hidden width is 2/3 of the
    plain MLP's so that the parameter count matches."""

    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        h = int(2 * hidden / 3)
        self.w12, self.w3, self.drop = nn.Linear(dim, 2 * h), nn.Linear(h, dim), nn.Dropout(drop)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.w3(self.drop(nn.functional.silu(a) * b))


class MoEFFN(nn.Module):
    """Sparse mixture-of-experts MLP: a router sends each vector to its top-k of E expert MLPs (gates renormalised over
    the k chosen). Train mode computes every expert densely with zero gates for the unchosen (compile-friendly, same
    result); eval mode only runs the chosen experts (CPU inference). `last_aux` holds the Switch load-balancing loss
    E * sum_e f_e * P_e (f_e: share of routing slots to expert e, P_e: mean router probability), 1 when balanced."""

    def __init__(self, dim, n_experts=8, topk=2, hidden_mult=2.0, drop=0.1):
        super().__init__()
        h = int(dim * hidden_mult)
        self.E, self.k = n_experts, topk
        self.router = nn.Linear(dim, n_experts, bias=False)
        nn.init.zeros_(self.router.weight)  # uniform routing at start
        self.w1 = nn.Parameter(torch.randn(n_experts, dim, h) * dim ** -0.5)
        self.b1 = nn.Parameter(torch.zeros(n_experts, h))
        self.w2 = nn.Parameter(torch.randn(n_experts, h, dim) * h ** -0.5 * 0.1)
        self.b2 = nn.Parameter(torch.zeros(n_experts, dim))
        self.drop = nn.Dropout(drop)
        self.last_aux = None

    def gates(self, x):
        probs = self.router(x).float().softmax(-1)                       # (N, E)
        topv, topi = probs.topk(self.k, -1)
        g = torch.zeros_like(probs).scatter(-1, topi, topv / topv.sum(-1, keepdim=True))
        f = torch.zeros_like(probs).scatter(-1, topi, 1.0).mean(0) / self.k
        self.last_aux = self.E * (f * probs.mean(0)).sum()
        return g.to(x.dtype), topi

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        g, topi = self.gates(x)
        if self.training:
            hid = self.drop(torch.nn.functional.gelu(torch.einsum("nd,edh->enh", x, self.w1) + self.b1[:, None]))
            y = torch.einsum("enh,ehd->end", hid, self.w2) + self.b2[:, None]   # (E, N, D)
            out = torch.einsum("ne,end->nd", g, y)
        else:
            out = torch.zeros_like(x)
            for e in range(self.E):
                rows = (topi == e).any(-1).nonzero(as_tuple=True)[0]
                if rows.numel():
                    hid = torch.nn.functional.gelu(x[rows] @ self.w1[e] + self.b1[e])
                    out.index_add_(0, rows, g[rows, e:e + 1] * (hid @ self.w2[e] + self.b2[e]))
        return out.reshape(shape)


# vertical centre (0 = top, 1 = bottom of the crop) of each attribute's region; None = whole body (gender, age)
ATTR_ROW = {0: None, 1: None, 2: None, 3: None, 4: .12, 5: .15, 6: .10, 7: .35, **{k: .35 for k in range(8, 20)},
            20: .65, **{k: .65 for k in range(21, 33)}, 33: .65, 34: .65, 35: .35, 36: .50, 37: .12, 38: .12, 39: .08}


def body_prior(grid, num_attrs, sigma=0.18):
    """(A, H*W) additive attention bias: -(y - c_k)^2 / (2 sigma^2) on the patch row y, 0 for whole-body attributes."""
    H, W = grid
    y = (torch.arange(H).float() + 0.5) / H
    out = torch.zeros(num_attrs, H, W)
    for k in range(num_attrs):
        c = ATTR_ROW.get(k)
        if c is not None:
            out[k] = (-(y - c) ** 2 / (2 * sigma ** 2))[:, None].expand(H, W)
    return out.reshape(num_attrs, H * W)


class TempMultiheadAttention(nn.MultiheadAttention):
    """nn.MultiheadAttention (same parameters, batch_first) whose attention logits of query a are divided by temp[a].
    temp > 1 flattens the softmax over the tokens: that query pools a wider region instead of one or two patches (the
    H+ query head reads clothing colours from ~1.4 tokens on average). Same call convention as nn.MultiheadAttention."""

    def __init__(self, dim, n_heads, temp, dropout=0.0):
        super().__init__(dim, n_heads, dropout=dropout, batch_first=True)
        self.register_buffer("temp", torch.as_tensor(temp, dtype=torch.float32), persistent=False)

    def forward(self, query, key, value, need_weights=False, attn_mask=None, average_attn_weights=True):
        B, A, D = query.shape
        N, h = key.shape[1], self.num_heads
        d = D // h
        wq, wk, wv = self.in_proj_weight.chunk(3)
        bq, bk, bv = self.in_proj_bias.chunk(3)
        q = nn.functional.linear(query, wq, bq).view(B, A, h, d).transpose(1, 2)
        k = nn.functional.linear(key, wk, bk).view(B, N, h, d).transpose(1, 2)
        v = nn.functional.linear(value, wv, bv).view(B, N, h, d).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) / (d ** 0.5) / self.temp.view(1, 1, A, 1).to(q.dtype)
        if attn_mask is not None:
            logits = logits + attn_mask.view(B, h, A, N).to(logits.dtype)
        w = logits.softmax(-1)
        out = (nn.functional.dropout(w, self.dropout, self.training) @ v).transpose(1, 2).reshape(B, A, D)
        out = self.out_proj(out)
        return (out, (w.mean(1) if average_attn_weights else w)) if need_weights else (out, None)


CLOTHING = range(7, 35)   # UpperBody-Length-Short .. LowerBody-Type-Skirt&Dress in the 40-attribute order


class EfficientCrossAttention(nn.Module):
    """Cross-attention of the attribute queries over the tokens without the softmax over tokens.

    linear:  out_i = phi(q_i)^T (sum_j phi(k_j) v_j^T) / (phi(q_i)^T sum_j phi(k_j)), phi = elu + 1 (Katharopoulos et
             al. 2020). Cost linear in the number of tokens; weights are positive but less peaked than a softmax.
    sigmoid: out_i = sum_j sigmoid(q_i.k_j / sqrt(d) - log n) v_j (Ramapuram et al. 2024). Every token is gated
             independently, so a query can take several regions at full weight instead of sharing one unit of mass.
    Same call convention as nn.MultiheadAttention(batch_first=True) for the parts QueryHead uses."""

    def __init__(self, dim, n_heads, kind="linear", drop=0.0):
        super().__init__()
        self.h, self.kind, self.d = n_heads, kind, dim // n_heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.drop = nn.Dropout(drop)

    def forward(self, query, key, value, need_weights=False, attn_mask=None):
        B, A, D = query.shape
        N = key.shape[1]
        q = self.q(query).view(B, A, self.h, self.d).transpose(1, 2)          # (B, h, A, d)
        k = self.k(key).view(B, N, self.h, self.d).transpose(1, 2)            # (B, h, N, d)
        v = self.v(value).view(B, N, self.h, self.d).transpose(1, 2)
        if self.kind == "linear":
            q, k = nn.functional.elu(q) + 1, nn.functional.elu(k) + 1
            kv = k.transpose(-2, -1) @ v                                       # (B, h, d, d)
            z = 1 / (q @ k.sum(2, keepdim=True).transpose(-2, -1) + 1e-6)      # (B, h, A, 1)
            out = (q @ kv) * z
        else:
            logits = q @ k.transpose(-2, -1) / self.d ** 0.5 - torch.log(torch.tensor(float(N)))
            if attn_mask is not None:
                logits = logits + attn_mask.view(B, self.h, A, N)
            out = self.drop(torch.sigmoid(logits)) @ v
        return self.o(out.transpose(1, 2).reshape(B, A, D)), None


@HEADS.register("query")
class QueryHead(nn.Module):
    supports_q = True  # forward(..., return_q=True) also returns the attribute embeddings (training-only decoders)

    """One learned query per attribute cross-attends to the tokens (ML-Decoder / Query2Label style).

    Each attribute can look at its own region (head for hats and glasses, legs for trousers), instead of
    sharing one pooled vector. Output of query a goes through its own linear classifier (group FC).
    """

    def __init__(self, dim, num_attrs, n_layers=1, n_heads=8, drop=0.1, mlp_ratio=2.0, self_attn=False,
                 n_domains=1, avg_domains=None, moe_experts=0, moe_topk=2, moe_where="tokens",
                 spatial_prior=False, fg_bias=False, grid=None, attn_type="softmax", mlp_type="gelu", attn_temp=1.0,
                 sep_values=False):
        super().__init__()
        # sep_values: keys and values come from different token sets (forward(..., values=...)); the values get their
        # own LayerNorm (joint model loc_read: DINOv3 decides where to look, the values also carry CLIP at that place)
        self.v_norm = nn.LayerNorm(dim) if sep_values else None
        # attention selectivity: a learnable body-part prior per attribute query and / or a data-dependent foreground
        # bias (CLS-patch similarity of the backbone) added to the attention logits over the patch tokens
        self.n_heads = n_heads
        self.prior = nn.Parameter(body_prior(grid, num_attrs)) if spatial_prior else None
        self.fg_w = nn.Parameter(torch.ones(n_heads)) if fg_bias else None
        self.fg_mode = fg_bias if isinstance(fg_bias, str) else "cls"   # fg_bias=true / "cls" or "pca" (spotlight)
        # mixture of experts: "tokens" routes every patch token to its experts before the attribute queries read them
        # (feature-type specific processing); "queries" replaces the query MLP (attribute-specific processing)
        self.tok_moe = None
        if moe_experts and "tokens" in moe_where:
            self.tok_moe_norm = nn.LayerNorm(dim)
            self.tok_moe = MoEFFN(dim, moe_experts, moe_topk, mlp_ratio, drop)
        q_moe = bool(moe_experts) and "queries" in moe_where
        self.self_attn, self.n_domains = self_attn, n_domains
        self.register_buffer("avg_domains", torch.tensor(avg_domains if avg_domains is not None else range(n_domains)),
                             persistent=False)
        self.queries = nn.Parameter(torch.randn(num_attrs, dim) * 0.02)
        self.kv_norm = nn.LayerNorm(dim)
        # attn_temp: softmax temperature of the clothing queries (colour, length, type); the others keep 1
        temp = torch.ones(num_attrs)
        if attn_temp != 1.0:
            assert num_attrs == 40, "attn_temp uses the 40-attribute order"
            temp[list(CLOTHING)] = attn_temp
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            extra = {"sa_norm": nn.LayerNorm(dim),  # attributes exchange information (Query2Label decoder)
                     "sa": nn.MultiheadAttention(dim, n_heads, dropout=drop, batch_first=True)} if self_attn else {}
            self.layers.append(nn.ModuleDict({
                **extra,
                "q_norm": nn.LayerNorm(dim),
                "attn": TempMultiheadAttention(dim, n_heads, temp, drop) if attn_type == "softmax" and attn_temp != 1.0
                else nn.MultiheadAttention(dim, n_heads, dropout=drop, batch_first=True) if attn_type == "softmax"
                else EfficientCrossAttention(dim, n_heads, attn_type, drop),
                "mlp_norm": nn.LayerNorm(dim),
                "mlp": MoEFFN(dim, moe_experts, moe_topk, mlp_ratio, drop) if q_moe else
                SwiGLU(dim, int(dim * mlp_ratio), drop) if mlp_type == "swiglu" else
                nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(drop),
                              nn.Linear(int(dim * mlp_ratio), dim)),
            }))
        self.out_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)
        if n_domains == 1:  # parameter names of the plain query head
            self.fc_weight = nn.Parameter(torch.randn(num_attrs, dim) * 0.02)
            self.fc_bias = nn.Parameter(torch.zeros(num_attrs))
        else:  # one classifier per source dataset: each keeps its own labelling convention
            self.fc_weight = nn.Parameter(torch.randn(n_domains, num_attrs, dim) * 0.02)
            self.fc_bias = nn.Parameter(torch.zeros(n_domains, num_attrs))

    def forward(self, tokens, n_prefix, ds=None, return_q=False, values=None):
        logits, q = self._logits(tokens, n_prefix, ds, values)
        return (logits, q) if return_q else logits

    def _logits(self, tokens, n_prefix, ds=None, values=None):
        t = torch.cat([tokens[:, :1], tokens[:, n_prefix:]], 1)  # CLS + patches, no registers
        if self.tok_moe is not None:
            t = t + self.tok_moe(self.tok_moe_norm(t))
        kv = self.kv_norm(t)
        vv = kv if values is None else self.v_norm(torch.cat([values[:, :1], values[:, n_prefix:]], 1))
        B, A = tokens.shape[0], self.queries.shape[0]
        mask = None
        if self.prior is not None or self.fg_w is not None:
            S = kv.shape[1]                                            # CLS + patches
            mask = kv.new_zeros(B, self.n_heads, A, S)
            if self.prior is not None:
                mask[..., 1:] = mask[..., 1:] + self.prior.to(kv.dtype)
            if self.fg_w is not None:
                c = nn.functional.normalize(t[:, :1].float(), dim=-1)
                pt = nn.functional.normalize(t[:, 1:].float(), dim=-1)
                sim = (pt @ c.transpose(1, 2)).squeeze(-1)             # (B, N) CLS-patch cosine similarity
                if self.fg_mode == "pca":
                    # PCA spotlight: projection of each patch on the image's first principal component (5 power
                    # iterations on the centred patch tokens). The first component separates person from background
                    # in DINO features; its sign is fixed so that it agrees with the CLS similarity.
                    X = t[:, 1:].float().detach()
                    X = X - X.mean(1, keepdim=True)
                    v = X.mean(1)[..., None] + 1e-3
                    v = torch.randn_like(v) if not torch.isfinite(v).all() else v
                    for _ in range(5):
                        v = nn.functional.normalize(X.transpose(1, 2) @ (X @ v), dim=1)
                    proj = (X @ v).squeeze(-1)                         # (B, N)
                    sgn = torch.sign(((proj - proj.mean(1, keepdim=True)) * (sim - sim.mean(1, keepdim=True))).sum(1, keepdim=True))
                    sim = proj * torch.where(sgn == 0, torch.ones_like(sgn), sgn)
                z = (sim - sim.mean(1, keepdim=True)) / (sim.std(1, keepdim=True) + 1e-6)
                mask[..., 1:] = mask[..., 1:] + (self.fg_w[None, :, None, None] * z[:, None, None, :]).to(kv.dtype)
            mask = mask.reshape(B * self.n_heads, A, S)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            if self.self_attn:
                h = layer["sa_norm"](q)
                q = q + layer["sa"](h, h, h, need_weights=False)[0]
            h = layer["q_norm"](q)
            q = q + layer["attn"](h, kv, vv, need_weights=False, attn_mask=mask)[0]
            q = q + layer["mlp"](layer["mlp_norm"](q))
        q = self.drop(self.out_norm(q))
        if self.n_domains == 1:
            return (q * self.fc_weight).sum(-1) + self.fc_bias, q
        if ds is None:  # unseen domain: average the conventions of the datasets trained on
            w, b = self.fc_weight[self.avg_domains].mean(0), self.fc_bias[self.avg_domains].mean(0)
            return (q * w).sum(-1) + b, q
        return (q * self.fc_weight[ds]).sum(-1) + self.fc_bias[ds], q


@HEADS.register("dual")
class DualHead(nn.Module):
    """Query head and linear head on the same tokens, logits blended: a two-model ensemble for one backbone pass.
    The linear head keeps the ranking of a global feature, the query head adds per-attribute spatial evidence."""

    def __init__(self, dim, num_attrs, alpha=0.5, **query_kw):
        super().__init__()
        self.alpha = alpha
        self.query = QueryHead(dim, num_attrs, **query_kw)
        self.linear = LinearHead(dim, num_attrs)

    def forward(self, tokens, n_prefix):
        return self.alpha * self.query(tokens, n_prefix) + (1 - self.alpha) * self.linear(tokens, n_prefix)


class _ConvNeXtBlock(nn.Module):
    def __init__(self, c, k=7, drop=0.1):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, padding=k // 2, groups=c)
        self.norm = nn.GroupNorm(1, c)  # = LayerNorm over (C, H, W), batch-size independent
        self.mlp = nn.Sequential(nn.Conv2d(c, 2 * c, 1), nn.GELU(), nn.Dropout(drop), nn.Conv2d(2 * c, c, 1))

    def forward(self, x):
        return x + self.mlp(self.norm(self.dw(x)))


@HEADS.register("conv")
class ConvHead(nn.Module):
    """CNN head on the ViT patch grid: patch tokens -> (B, D, H, W) -> 1x1 reduce -> ConvNeXt blocks -> CSRA pooling.

    CSRA (Zhu & Wu, ICCV 2021): each attribute gets a spatial score map; its logit is the mean score plus lam times
    an attention-weighted score, so small attributes (hat, glasses, shoes) come from their strongest region instead
    of being diluted by global averaging. The CLS token is added to every cell as global context."""

    def __init__(self, dim, num_attrs, grid=(16, 8), width=256, n_blocks=2, lam=0.3, temp=1.0, drop=0.1):
        super().__init__()
        self.grid, self.lam, self.temp = tuple(grid), lam, temp
        self.in_norm = nn.LayerNorm(dim)
        self.reduce = nn.Conv2d(dim, width, 1)
        self.blocks = nn.Sequential(*[_ConvNeXtBlock(width, drop=drop) for _ in range(n_blocks)])
        self.out_norm = nn.GroupNorm(1, width)
        self.drop = nn.Dropout2d(drop)
        self.score = nn.Conv2d(width, num_attrs, 1)

    def forward(self, tokens, n_prefix):
        t = self.in_norm(tokens)
        cls, patches = t[:, :1], t[:, n_prefix:]
        H, W = self.grid
        B, N, D = patches.shape
        assert N == H * W, f"{N} patch tokens do not fit the grid {self.grid}"
        x = (patches + cls).transpose(1, 2).reshape(B, D, H, W)
        x = self.drop(self.out_norm(self.blocks(self.reduce(x))))
        s = self.score(x).flatten(2)                     # (B, A, H*W) per-attribute score maps
        att = (s * self.temp).softmax(-1)
        return s.mean(-1) + self.lam * (att * s).sum(-1)


# attributes from global to local, top to bottom: gender, age, hair, hat, glasses, upper length / colour,
# lower length / type / colour, backpack, bag. Groups are read one after the other by the sequential heads.
SEQ_GROUPS = [[3], [0, 1, 2], [4, 5, 6], [39], [37, 38], [7], list(range(8, 20)), [20], [33, 34],
              list(range(21, 33)), [35], [36]]


class SelectiveSSM(nn.Module):
    """Mamba (S6) block over a sequence, causal: h_t = exp(dt_t * A) h_{t-1} + dt_t B_t u_t, y_t = C_t h_t + D u_t,
    with dt, B, C computed from the input (selective: each step decides how much of the past to keep), gated by
    SiLU(z), short causal depthwise conv in front (Gu & Dao 2023). A linear recurrence, so it can also be evaluated
    with a parallel scan; the sequences here are short (<= 40 steps) and use the plain loop."""

    def __init__(self, dim, d_inner=512, d_state=16, d_conv=4):
        super().__init__()
        self.d_inner, self.d_state = d_inner, d_state
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, 1 + 2 * d_state)  # dt (shared rank-1), B, C
        self.dt_proj = nn.Linear(1, d_inner)
        nn.init.constant_(self.dt_proj.bias, -2.0)  # dt ~ softplus(-2) = 0.13 at start
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, dim)
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)  # starts as identity (residual)

    def forward(self, x):  # (B, L, dim) -> (B, L, dim), causal along L
        Bsz, L, _ = x.shape
        u, z = self.in_proj(self.norm(x)).chunk(2, -1)
        u = nn.functional.silu(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))
        dt_r, Bm, Cm = self.x_proj(u).split([1, self.d_state, self.d_state], -1)
        dt = nn.functional.softplus(self.dt_proj(dt_r)).float()                 # (B, L, d_inner)
        A = -torch.exp(self.A_log.float())                                       # (d_inner, N)
        h = x.new_zeros(Bsz, self.d_inner, self.d_state, dtype=torch.float32)
        ys = []
        for t in range(L):
            h = torch.exp(dt[:, t, :, None] * A) * h + (dt[:, t] * u[:, t].float())[..., None] * Bm[:, t, None].float()
            ys.append((h * Cm[:, t, None].float()).sum(-1))
        y = (torch.stack(ys, 1) + u.float() * self.D).to(x.dtype) * nn.functional.silu(z)
        return x + self.out_proj(y)


@HEADS.register("seq")
class SequentialHead(nn.Module):
    """Attribute queries read the image, then information flows through the attributes in SEQ_GROUPS order.

    mode="mamba": all queries cross-attend in parallel, then a selective SSM scans the 40 attribute features in
      order (parallel reading, sequential reasoning).
    mode="seqattn": groups read the image one after another; a selective gated state carries what the previous
      groups found and is added to the next group's queries before they attend, so where a group looks depends
      on what was found before (age, then hair, ...). Within a group the queries work in parallel.
    Only hidden states are passed on, never predicted labels (no exposure bias, no hard co-occurrence rules)."""

    def __init__(self, dim, num_attrs, mode="mamba", n_heads=8, drop=0.1, mlp_ratio=2.0, d_state=16, d_inner=512):
        super().__init__()
        assert sorted(sum(SEQ_GROUPS, [])) == list(range(num_attrs))
        self.mode = mode
        self.order = [a for g in SEQ_GROUPS for a in g]
        self.register_buffer("inv_order", torch.argsort(torch.tensor(self.order)), persistent=False)
        self.queries = nn.Parameter(torch.randn(num_attrs, dim) * 0.02)
        self.kv_norm = nn.LayerNorm(dim)
        self.q_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=drop, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(int(dim * mlp_ratio), dim))
        if mode == "mamba":
            self.ssm = SelectiveSSM(dim, d_inner, d_state)
        elif mode == "seqattn":
            self.state_in = nn.Linear(dim, dim)        # group summary -> candidate state
            self.state_gate = nn.Linear(2 * dim, dim)  # selective forget gate from (state, summary)
            self.state_out = nn.Linear(dim, dim)       # state -> query shift for the next group
            nn.init.zeros_(self.state_out.weight); nn.init.zeros_(self.state_out.bias)
        else:
            raise ValueError(mode)
        self.out_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)
        self.fc_weight = nn.Parameter(torch.randn(num_attrs, dim) * 0.02)
        self.fc_bias = nn.Parameter(torch.zeros(num_attrs))

    def read(self, q, kv):
        q = q + self.attn(self.q_norm(q), kv, kv, need_weights=False)[0]
        return q + self.mlp(self.mlp_norm(q))

    def forward(self, tokens, n_prefix):
        kv = self.kv_norm(torch.cat([tokens[:, :1], tokens[:, n_prefix:]], 1))
        B = tokens.shape[0]
        if self.mode == "mamba":
            q = self.read(self.queries[self.order].unsqueeze(0).expand(B, -1, -1), kv)
            q = self.ssm(q)[:, self.inv_order]                      # back to attribute order
        else:
            state = tokens.new_zeros(B, self.queries.shape[1])
            outs = [None] * len(SEQ_GROUPS)
            for gi, g in enumerate(SEQ_GROUPS):
                qg = self.queries[g].unsqueeze(0).expand(B, -1, -1) + self.state_out(state)[:, None]
                qg = self.read(qg, kv)
                summary = qg.mean(1)
                gate = torch.sigmoid(self.state_gate(torch.cat([state, summary], -1)))
                state = gate * state + (1 - gate) * torch.tanh(self.state_in(summary))
                outs[gi] = qg
            q = torch.cat(outs, 1)[:, self.inv_order]
        q = self.drop(self.out_norm(q))
        return (q * self.fc_weight).sum(-1) + self.fc_bias


@HEADS.register("joint")
class JointHead(nn.Module):
    """Full self-attention over [40 attribute queries ; CLS + patch tokens]: queries read the tokens and the tokens are
    updated by the queries and by each other (pre-norm transformer layers), instead of cross-attention only."""
    supports_q = True

    def __init__(self, dim, num_attrs, n_layers=1, n_heads=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_attrs, dim) * 0.02)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(dim, n_heads, int(dim * mlp_ratio), drop, activation="gelu",
                                                                batch_first=True, norm_first=True) for _ in range(n_layers)])
        self.out_norm, self.drop = nn.LayerNorm(dim), nn.Dropout(drop)
        self.fc_weight = nn.Parameter(torch.randn(num_attrs, dim) * 0.02)
        self.fc_bias = nn.Parameter(torch.zeros(num_attrs))

    def forward(self, tokens, n_prefix, return_q=False):
        A = self.queries.shape[0]
        x = torch.cat([self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1), tokens[:, :1], tokens[:, n_prefix:]], 1)
        for layer in self.layers:
            x = layer(x)
        q = self.drop(self.out_norm(x[:, :A]))
        logits = (q * self.fc_weight).sum(-1) + self.fc_bias
        return (logits, q) if return_q else logits


@HEADS.register("inject")
class InjectHead(nn.Module):
    """Attribute tokens inside the backbone: 40 learned tokens are inserted after the prefix tokens before the last k
    ViT blocks (PARModel handles the insertion) and take part in full self-attention with the patches there. Each
    attribute is read from its own token by its own classifier. The tokens live in the head so that they get the
    head learning rate."""
    supports_q = True

    def __init__(self, dim, num_attrs, k=4, drop=0.1):
        super().__init__()
        self.k = k
        self.attr_tokens = nn.Parameter(torch.randn(num_attrs, dim) * 0.02)
        self.out_norm, self.drop = nn.LayerNorm(dim), nn.Dropout(drop)
        self.fc_weight = nn.Parameter(torch.randn(num_attrs, dim) * 0.02)
        self.fc_bias = nn.Parameter(torch.zeros(num_attrs))

    def forward(self, tokens, n_prefix, return_q=False):
        A = self.attr_tokens.shape[0]
        q = self.drop(self.out_norm(tokens[:, n_prefix:n_prefix + A]))
        logits = (q * self.fc_weight).sum(-1) + self.fc_bias
        return (logits, q) if return_q else logits


@HEADS.register("inject_joint")
class InjectJointHead(InjectHead):
    """Attribute tokens inside the last k backbone blocks (InjectHead) followed by n full self-attention layers over
    [attribute tokens ; CLS + patches] (JointHead): backbone-level attribute latents, plus a head that can still
    re-read the patches."""

    def __init__(self, dim, num_attrs, k=4, n_layers=1, n_heads=8, mlp_ratio=4.0, drop=0.1):
        super().__init__(dim, num_attrs, k, drop)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(dim, n_heads, int(dim * mlp_ratio), drop, activation="gelu",
                                                                batch_first=True, norm_first=True) for _ in range(n_layers)])

    def forward(self, tokens, n_prefix, return_q=False):
        A = self.attr_tokens.shape[0]
        x = torch.cat([tokens[:, n_prefix:n_prefix + A], tokens[:, :1], tokens[:, n_prefix + A:]], 1)
        for layer in self.layers:
            x = layer(x)
        q = self.drop(self.out_norm(x[:, :A]))
        logits = (q * self.fc_weight).sum(-1) + self.fc_bias
        return (logits, q) if return_q else logits


class TextAnchor:
    """Language-anchored path for CLIP backbones, mixed into a head (see QueryTextHead).

    The CLS token is mapped into CLIP's image-text space by the frozen pretrained projection P and scored against a
    per-attribute text direction w_k = t(attribute prompt) - t("a photo of a person"), the CLIP zero-shot classifier:
        text_logit_k = s * <normalize(P cls), w0_k + delta_scale * delta_k> + b_k
    w0 (text) and P are buffers filled by init_text() before training (and restored from the checkpoint afterwards);
    delta starts at 0 and moves with a delta_scale-times smaller effective step than the rest of the head, so the
    classifier of a rare attribute stays near its language description instead of fitting the source datasets alone.
    """

    def _add_text(self, dim, num_attrs, embed_dim=768, scale=100.0, delta_scale=0.1):
        self.register_buffer("proj", torch.zeros(dim, embed_dim))
        self.register_buffer("text_w0", torch.zeros(num_attrs, embed_dim))
        self.text_delta = nn.Parameter(torch.zeros(num_attrs, embed_dim))
        self.text_b = nn.Parameter(torch.zeros(num_attrs))
        self.scale, self.delta_scale = scale, delta_scale

    @torch.no_grad()
    def init_text(self, path, prior):
        d = torch.load(path, map_location="cpu")
        self.proj.copy_(d["proj"]); self.text_w0.copy_(d["pos"] - d["neutral"][None])
        p = torch.as_tensor(prior, dtype=torch.float32).clamp(1e-3, 1 - 1e-3)
        self.text_b.copy_(torch.log(p / (1 - p)))

    def text_logits(self, tokens):
        e = nn.functional.normalize(tokens[:, 0].float() @ self.proj, dim=-1)
        w = self.text_w0 + self.delta_scale * self.text_delta
        return self.scale * e @ w.T + self.text_b


@HEADS.register("query_text")
class QueryTextHead(QueryHead, TextAnchor):
    """Query head + the language-anchored CLS path (TextAnchor): logit = query_k(tokens) + text_logit_k."""

    def __init__(self, dim, num_attrs, embed_dim=768, scale=100.0, delta_scale=0.1, **query_kw):
        super().__init__(dim, num_attrs, **query_kw)
        self._add_text(dim, num_attrs, embed_dim, scale, delta_scale)

    def forward(self, tokens, n_prefix, ds=None, return_q=False):
        logits, q = self._logits(tokens, n_prefix, ds)
        logits = logits + self.text_logits(tokens).to(logits.dtype)
        return (logits, q) if return_q else logits


@HEADS.register("inject_text")
class InjectTextHead(InjectHead, TextAnchor):
    """Attribute tokens inside the last k blocks (InjectHead) + the language-anchored CLS path (TextAnchor)."""

    def __init__(self, dim, num_attrs, k=4, drop=0.1, embed_dim=768, scale=100.0, delta_scale=0.1):
        super().__init__(dim, num_attrs, k, drop)
        self._add_text(dim, num_attrs, embed_dim, scale, delta_scale)

    def forward(self, tokens, n_prefix, return_q=False):
        logits, q = super().forward(tokens, n_prefix, return_q=True)
        logits = logits + self.text_logits(tokens).to(logits.dtype)
        return (logits, q) if return_q else logits


@HEADS.register("query_text_region")
class QueryTextRegionHead(QueryTextHead):
    """QueryTextHead + region anchors for the clothing colours (CLIP backbones).

    Colour words are scored where the garment is: the upper-body colours on the upper rows of the patch grid, the
    lower-body colours on the lower rows. The region embedding is the MaskCLIP dense embedding of each patch (value path of
    the last attention block, final norm, frozen CLIP projection; PARModel puts these patch tokens in self.dense before
    calling the head), normalised per patch, averaged over the rows and normalised again. Its score against the same
    text direction as the CLS anchor is added:  logit_k += s * <e_region(k), w_k>  for the 24 colour attributes.
    """

    def __init__(self, dim, num_attrs, grid=None, up_rows=(2, 8), lo_rows=(8, 15), upper=(8, 20), lower=(21, 33), **kw):
        super().__init__(dim, num_attrs, **kw)
        self.grid, self.up_rows, self.lo_rows = tuple(grid), tuple(up_rows), tuple(lo_rows)
        self.upper, self.lower = list(range(*upper)), list(range(*lower))
        self.dense = None

    def region_logits(self):
        h, w = self.grid
        with torch.autocast(self.dense.device.type, enabled=False):  # float32 throughout (index assignment below)
            e = nn.functional.normalize(self.dense.float() @ self.proj.float(), dim=-1)
            g = e.reshape(e.shape[0], h, w, -1)
            up = nn.functional.normalize(g[:, self.up_rows[0]:self.up_rows[1]].mean((1, 2)), dim=-1)
            lo = nn.functional.normalize(g[:, self.lo_rows[0]:self.lo_rows[1]].mean((1, 2)), dim=-1)
            wt = (self.text_w0 + self.delta_scale * self.text_delta).float()
            out = e.new_zeros(e.shape[0], wt.shape[0])
            out[:, self.upper] = self.scale * up @ wt[self.upper].T
            out[:, self.lower] = self.scale * lo @ wt[self.lower].T
        self.dense = None
        return out

    def forward(self, tokens, n_prefix, ds=None, return_q=False):
        logits, q = self._logits(tokens, n_prefix, ds)
        logits = logits + (self.text_logits(tokens) + self.region_logits()).to(logits.dtype)
        return (logits, q) if return_q else logits


# slot roles of the region-slot head and the slot each attribute starts reading from (40-attribute order)
_HEAD_S, _UP_S, _LO_S, _FEET_S, _CARRY_S, _BG_S = range(6)
_ATTR_SLOT = ([None] * 4 + [_HEAD_S] * 3 + [_UP_S] * 13 + [_LO_S] * 15 + [_CARRY_S] * 2 + [_HEAD_S] * 3)
_COLOURS = ((_UP_S, list(range(8, 19)), 19), (_LO_S, list(range(21, 32)), 32))   # (slot, named colours, "Other")


@HEADS.register("rscg")
class RegionSlotColourHead(nn.Module):
    """Region-slot colour-geometry head (joint head of the dual model, --dual-kw joint_head=rscg).

    1. Region slots. Six learned slots (head, upper body, lower body, feet, carried objects, background) compete for
       the tokens: each token's assignment is a softmax over the slots (slot attention, Locatello et al. 2020), plus a
       learned per-slot bias over token positions. Competition partitions the image into regions instead of letting
       every query attend anywhere.
    2. Region-gated attribute queries. Attribute a has a learned mixture pi_a over the slots (initialised on its body
       region) and its cross-attention logits get log(sum_s pi_as M_sn): the query reads only inside its region.
    3. Colour geometry. Each garment slot emits n_codes colour codes in a code_dim space, measured relative to a code
       of the background slot (r = u - e, the scene illuminant / exposure cancels). Named colour c adds
       logsumexp_k(-|r_k - mu_c|^2 / 2 sigma_c^2) + b_c, so adjacent colours (black, grey, white) share one axis;
       "Other" adds g (softmin distance to all prototypes - rho): a colour far from every named one.
    """

    # v2 (cat=True, grids given): each colour code is a categorical distribution over the 11 named colours plus
    # "Other" (a learned constant logit, open-set style), so colours compete instead of firing independently; a garment
    # with K codes has p_c = 1 - prod_k (1 - p_kc) ("some code picks c") and adds lam_g * log p_c. Slot position bias
    # starts at a body-row prior (head / upper / lower / feet / carried rows), the same partition for every seed.
    SLOT_ROW = ((.10, .10), (.35, .15), (.65, .15), (.92, .08), (.50, .30), None)   # (centre, sigma) per slot

    def __init__(self, dim, num_attrs, n_tok=228, n_slots=6, iters=2, n_heads=8, drop=0.1, mlp_ratio=2.0, code_dim=8,
                 n_codes=2, cat=False, grids=None, codes=None, **_):
        super().__init__()
        assert num_attrs == 40 and n_slots == 6, "rscg uses the 40-attribute order and six region slots"
        self.iters, self.n_heads, self.n_codes, self.code_dim, self.cat = iters, n_heads, n_codes, code_dim, bool(cat)
        self.codes = tuple(codes) if codes else (n_codes,) * len(_COLOURS)
        self.slots0 = nn.Parameter(torch.randn(n_slots, dim) * 0.02)
        self.pos_bias = nn.Parameter(self.slot_prior(grids, n_slots, n_tok) if grids else torch.zeros(n_slots, n_tok))
        self.in_norm, self.s_norm, self.s_mlp_norm = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.sk, self.sv, self.sq, self.s_out = (nn.Linear(dim, dim) for _ in range(4))
        self.s_mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        gate = torch.zeros(num_attrs, n_slots)
        for a, s in enumerate(_ATTR_SLOT):
            if s is not None:
                gate[a, s] = 3.0
        gate[:, _BG_S] -= 3.0
        self.slot_gate = nn.Parameter(gate)
        self.queries = nn.Parameter(torch.randn(num_attrs, dim) * 0.02)
        self.kv_norm, self.q_norm, self.mlp_norm = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=drop, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(int(dim * mlp_ratio), dim))
        self.out_norm, self.drop = nn.LayerNorm(dim), nn.Dropout(drop)
        self.fc_weight = nn.Parameter(torch.randn(num_attrs, dim) * 0.02)
        self.fc_bias = nn.Parameter(torch.zeros(num_attrs))
        self.code = nn.ModuleList(nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, k * code_dim)) for k in self.codes)
        if self.cat:
            self.other_logit = nn.Parameter(torch.zeros(len(_COLOURS)))
            self.lam = nn.Parameter(torch.ones(len(_COLOURS)))
        self.illum = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, code_dim))
        n_col = len(_COLOURS[0][1])
        self.proto = nn.Parameter(torch.randn(len(_COLOURS), n_col, code_dim) * 0.5)
        self.log_sigma = nn.Parameter(torch.zeros(len(_COLOURS), n_col))
        self.c_bias = nn.Parameter(torch.zeros(len(_COLOURS), n_col))
        self.other_rho = nn.Parameter(torch.full((len(_COLOURS),), 3.0))
        self.other_gain = nn.Parameter(torch.ones(len(_COLOURS)))
        self.last_assign = None   # (B, S, N) slot assignment of the last batch, for inspection

    def forward(self, tokens, n_prefix, values=None):
        x = self.in_norm(tokens)
        B, N, D = x.shape
        k, v = self.sk(x), self.sv(x)
        s = self.slots0.unsqueeze(0).expand(B, -1, -1)
        for _ in range(self.iters):
            q = self.sq(self.s_norm(s))
            logit = (q @ k.transpose(1, 2)).float() / D ** 0.5 + self.pos_bias[:, :N].float()
            M = torch.softmax(logit, dim=1)                                  # (B, S, N): tokens compete for slots
            w = M / (M.sum(-1, keepdim=True) + 1e-6)
            s = s + self.s_out((w.to(v.dtype) @ v))
            s = s + self.s_mlp(self.s_mlp_norm(s))
        self.last_assign = M.detach()
        region = torch.softmax(self.slot_gate.float(), -1) @ M               # (B, A, N): where each attribute may look
        mask = torch.log(region + 1e-4)
        kv = self.kv_norm(tokens)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        h = self.q_norm(q)
        mask = mask.repeat_interleave(self.n_heads, 0).to(h.dtype)
        q = q + self.attn(h, kv, kv, need_weights=False, attn_mask=mask)[0]
        q = q + self.mlp(self.mlp_norm(q))
        q = self.drop(self.out_norm(q))
        logits = ((q * self.fc_weight).sum(-1) + self.fc_bias).float()
        e = self.illum(s[:, _BG_S]).float()                                   # (B, code_dim) scene code
        add = torch.zeros_like(logits)
        for g, (slot, named, other) in enumerate(_COLOURS):
            u = self.code[g](s[:, slot]).float().view(B, self.codes[g], self.code_dim) - e[:, None]
            if self.cat:
                d2 = ((u[:, :, None] - self.proto[g].float()[None, None]) ** 2).sum(-1) * torch.exp(-2 * self.log_sigma[g].float())
                z = torch.cat([-d2 / 2 + self.c_bias[g].float(), self.other_logit[g].float().expand(B, self.codes[g], 1)], -1)
                p = 1 - torch.prod(1 - torch.softmax(z, -1), 1)                                 # (B, C + 1)
                lp = torch.log(p.clamp_min(1e-6)) * self.lam[g].float()
                add[:, named] = lp[:, :-1]
                add[:, other] = lp[:, -1]
                continue
            d2 = ((u[:, :, None] - self.proto[g].float()[None, None]) ** 2).sum(-1)       # (B, K, C)
            d2 = d2 * torch.exp(-2 * self.log_sigma[g].float())
            add[:, named] = torch.logsumexp(-d2 / 2, 1) + self.c_bias[g].float()
            soft_min = -torch.logsumexp(-d2.flatten(1), 1)                                # (B,)
            add[:, other] = self.other_gain[g].float() * (soft_min - self.other_rho[g].float())
        return logits + add

    @classmethod
    def slot_prior(cls, grids, n_slots, n_tok):
        """(S, N) bias over [CLS, grid 1 patches, CLS, grid 2 patches]: -(y - c)^2 / 2 sigma^2 on the patch row y."""
        rows = []
        for h, w in grids:
            y = ((torch.arange(h).float() + 0.5) / h)[:, None].expand(h, w).reshape(-1)
            rows += [torch.full((1,), float("nan")), y]
        y = torch.cat(rows)
        assert len(y) == n_tok, (len(y), n_tok)
        out = torch.zeros(n_slots, n_tok)
        for s, cs in enumerate(cls.SLOT_ROW):
            if cs is not None:
                out[s] = torch.nan_to_num(-(y - cs[0]) ** 2 / (2 * cs[1] ** 2), nan=0.0).clamp_min(-6.0)   # learnable, not a hard ban
        return out
