"""Training-only reconstruction decoders: an auxiliary loss on the backbone tokens / attribute embeddings, dropped
after training (not part of the model, its checkpoints or the submission).

Target: the input image average-pooled to 4x4 cells per 16x16 patch (48 values per patch, in the normalised input
space). Not normalised per patch as in MAE: the patch mean is the colour, and 24 of the 40 attributes are colours.

  mae:  75% of the patch tokens are hidden; a small transformer sees the 25% left (+ mask tokens and positions) and
        reconstructs the hidden patches -> every token must carry appearance / context of its neighbourhood.
  attr: patch-position queries cross-attend to the 40 attribute embeddings only and reconstruct the whole image
        -> the attribute embeddings must describe the person's actual appearance (analysis by synthesis).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def patch_targets(img, grid, patch=16, sub=4):
    """(B, 3, H*patch, W*patch) -> (B, H*W, 3*sub*sub) average-pooled cells of each patch."""
    H, W = grid
    x = F.avg_pool2d(img.float(), patch // sub)                        # (B, 3, H*sub, W*sub)
    B = x.shape[0]
    x = x.reshape(B, 3, H, sub, W, sub).permute(0, 2, 4, 3, 5, 1)      # (B, H, W, sub, sub, 3)
    return x.reshape(B, H * W, sub * sub * 3)


class MaskedPatchDecoder(nn.Module):
    def __init__(self, dim, grid, width=256, depth=2, heads=4, mask_ratio=0.75, sub=4):
        super().__init__()
        self.grid, self.mask_ratio, self.sub = tuple(grid), mask_ratio, sub
        n = grid[0] * grid[1]
        self.embed = nn.Linear(dim, width)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos = nn.Parameter(torch.randn(1, n, width) * 0.02)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.0, batch_first=True,
                                                                norm_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.pred = nn.Linear(width, 3 * sub * sub)

    def forward(self, patches, img):
        B, N, _ = patches.shape
        target = patch_targets(img, self.grid, sub=self.sub)
        n_keep = max(1, int(round(N * (1 - self.mask_ratio))))
        keep = torch.rand(B, N, device=patches.device).argsort(1)[:, :n_keep]
        vis = torch.gather(self.embed(patches), 1, keep[..., None].expand(-1, -1, self.pos.shape[-1]))
        h = self.mask_token.expand(B, N, -1).to(vis.dtype).scatter(1, keep[..., None].expand_as(vis), vis) + self.pos
        for blk in self.blocks:
            h = blk(h)
        pred = self.pred(self.norm(h)).float()
        hidden = torch.ones(B, N, dtype=torch.bool, device=patches.device).scatter(1, keep, False)
        return ((pred - target) ** 2).mean(-1)[hidden].mean()


class AttributeToImageDecoder(nn.Module):
    def __init__(self, dim, grid, width=256, depth=2, heads=4, sub=4):
        super().__init__()
        self.grid, self.sub = tuple(grid), sub
        self.mem = nn.Linear(dim, width)
        self.pos_q = nn.Parameter(torch.randn(1, grid[0] * grid[1], width) * 0.02)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(width, heads, 2 * width, dropout=0.0, batch_first=True,
                                                                norm_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.pred = nn.Linear(width, 3 * sub * sub)

    def forward(self, attr_emb, img):
        target = patch_targets(img, self.grid, sub=self.sub)
        mem = self.mem(attr_emb)
        h = self.pos_q.expand(attr_emb.shape[0], -1, -1).to(mem.dtype)
        for layer in self.layers:
            h = layer(h, mem)
        return ((self.pred(self.norm(h)).float() - target) ** 2).mean()


AUX_DECODERS = {"mae": MaskedPatchDecoder, "attr": AttributeToImageDecoder}
