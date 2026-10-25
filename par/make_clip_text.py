"""Text anchors for the query_text head: OpenAI CLIP ViT-L/14 projection + one prompt-ensembled text embedding per
attribute and a neutral one ("a person"), saved for train.py --text-init.

python make_clip_text.py upar/models/clip_l14_text.pt   (needs open_clip_torch; checks that timm's CLS @ proj equals
open_clip's image embedding and prints the zero-shot AUC of the anchors on the val images of each dataset)
"""
import sys

import numpy as np
import open_clip
import timm
import torch
from PIL import Image

from upar.data import attr_names, build_transforms, load_split, UPARDataset

COLOURS = ["black", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "white", "yellow"]
PROMPTS = (["a young person, a teenager or young adult", "a middle-aged adult", "an elderly old person",
            "a woman", "a person with short hair", "a person with long hair", "a bald person",
            "a person wearing a short-sleeved top"]
           + [f"a person wearing a {c} top" for c in COLOURS] + ["a person wearing a multicoloured patterned top"]
           + ["a person wearing shorts or a short skirt"]
           + [f"a person wearing {c} trousers" for c in COLOURS] + ["a person wearing multicoloured patterned trousers"]
           + ["a person wearing long trousers or shorts", "a person wearing a skirt or a dress",
              "a person carrying a backpack", "a person carrying a handbag or shoulder bag",
              "a person wearing eyeglasses", "a person wearing sunglasses", "a person wearing a hat or a cap"])
NEUTRAL = "a person"
TEMPLATES = ["a photo of {}.", "a low resolution surveillance photo of {}.", "a cropped photo of {}, full body.",
             "a blurry photo of {}."]


@torch.no_grad()
def encode(model, tok, texts):
    out = []
    for t in texts:
        e = model.encode_text(tok([tp.format(t) for tp in TEMPLATES]).cuda()).float()
        e = torch.nn.functional.normalize(e, dim=-1).mean(0)
        out.append(torch.nn.functional.normalize(e, dim=0))
    return torch.stack(out)


def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def main(out):
    A = attr_names(); assert len(PROMPTS) == len(A)
    model, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = model.cuda().eval(); tok = open_clip.get_tokenizer("ViT-L-14")
    pos, neutral = encode(model, tok, PROMPTS), encode(model, tok, [NEUTRAL])[0]
    proj = model.visual.proj.detach().float().cpu()                      # (1024, 768)
    torch.save({"proj": proj, "pos": pos.cpu(), "neutral": neutral.cpu(), "prompts": PROMPTS}, out)

    # timm tokens -> same embedding as open_clip at 224x224 (sanity check of the projection path)
    tm = timm.create_model("vit_large_patch14_clip_224.openai", pretrained=True, num_classes=0).cuda().eval()
    va = load_split("val")
    ds = UPARDataset("val", build_transforms((224, 224), False, mean=pre.transforms[-1].mean, std=pre.transforms[-1].std))
    x = torch.stack([ds[i][0] for i in range(8)]).cuda()
    with torch.no_grad():
        a = torch.nn.functional.normalize(tm.forward_features(x)[:, 0].float() @ proj.cuda(), dim=-1)
        b = torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1)
    print("timm vs open_clip image embedding cosine:", (a * b).sum(-1).min().item())

    # zero-shot AUC of the anchors at the training input size (224x112), per dataset
    tm = timm.create_model("vit_large_patch14_clip_224.openai", pretrained=True, num_classes=0, img_size=(224, 112)).cuda().eval()
    ds = UPARDataset("val", build_transforms((224, 112), False, mean=pre.transforms[-1].mean, std=pre.transforms[-1].std))
    w = (pos - neutral[None]).cuda()
    for name in ("PETA", "Market1501", "PA100k"):
        rows = np.where(va.ds.values == name)[0][:3000]
        sc = []
        with torch.no_grad():
            for i in range(0, len(rows), 128):
                x = torch.stack([ds[j][0] for j in rows[i:i + 128]]).cuda()
                e = torch.nn.functional.normalize(tm.forward_features(x)[:, 0].float() @ proj.cuda(), dim=-1)
                sc.append((e @ w.T).cpu().numpy())
        sc = np.concatenate(sc); y = va[A].values[rows]
        aucs = [auc(y[:, k], sc[:, k]) for k in range(len(A)) if 0 < y[:, k].sum() < len(y)]
        print(f"zero-shot AUC {name}: {np.mean(aucs):.4f} over {len(aucs)} attributes")


if __name__ == "__main__":
    main(sys.argv[1])
