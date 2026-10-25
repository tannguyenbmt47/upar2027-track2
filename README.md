# UPAR Challenge 2027, Track 2 (Attribute-based Person Retrieval)

Code, training scripts and trained weights of our Track 2 leaderboard entry to the
[UPAR Challenge 2027](https://vap.aau.dk/rws/challenge/) (Real-World Surveillance workshop, WACV 2027).
Weights: [huggingface.co/tannguyenbmt47/upar2027-track2](https://huggingface.co/tannguyenbmt47/upar2027-track2).

| | v13: leaderboard entry |
|---|---|
| Hidden test set (Codabench) | **mADM 0.6817** |
| Submission zip | `t2_stu_hpG_clipTGw70_rerank_gpu_v13.zip` |
| Models | DINOv3 ViT-L/16 student, DINOv3 ViT-H+/16 and CLIP ViT-L/14 (text score, WiSE-FT 0.7), logits averaged |
| Ranking | kNN smoothing of the gallery logits (k = 50, alpha = 0.2), then the log-likelihood of the exact query |
| Weights | `student.pt`, `hp.pt`, `m1.pt` |
| Build / train | `submission/build_v13.py`, `scripts/train_v13.sh` |

## Contents

| Path | What it is |
|---|---|
| `par/train.py` | training script (every option used by the submitted models is a command-line flag) |
| `par/upar/` | data loading, augmentation, losses (BCE with DeepMAR weights, group cross-entropy, distillation), metrics, retrieval scores and models (`upar/models/__init__.py`, query heads in `upar/models/heads.py`) |
| `par/upar/models/clip_l14_text.pt` | the 40 CLIP attribute text vectors and the CLIP projection used by the text score |
| `par/make_clip_text.py` | rebuilds `clip_l14_text.pt` from the frozen CLIP text encoder (needs `open_clip_torch`) |
| `par/make_wise_member.py` | WiSE-FT of a finished run: encoder weights (1 − α) · pretrained + α · fine-tuned (α = 0.7) |
| `par/export_member.py` | exports the EMA weights of a finished run as bf16 plus its config |
| `par/eval_retrieval.py` | Track 2 metrics (mADM, mAP, Rank-k) of saved validation logits, one gallery per dataset |
| `submission/` | the files of the Codabench submission except the weights: `run.py`, `metadata.yaml`, `vendor/` (timm 1.0.30 and the model code), `v13/config.json` |
| `submission/build_v13.py` | zips the files above with the weights |
| `scripts/train_v13.sh` | trains, exports and packages all models of v13 |
| `scripts/check_t2.py` | runs a zip the way the Codabench ingestion program does, on one validation gallery |
| `scripts/upload_hf.py` | uploads the weights to Hugging Face |
| `Dockerfile`, `Dockerfile.inference` | training image (CUDA 12.8) and the image used by the grader (PyTorch 2.4.1, CUDA 12.1) |

## 1. Data

Download the challenge data with the organizers' starter kit into `UPAR-Challenge-2027/` at the root of this
repository:

```bash
git clone https://github.com/speckean/UPAR-Challenge-2027
cd UPAR-Challenge-2027 && python download_datasets.py --data-dir ./data && cd ..
```

The code expects `UPAR-Challenge-2027/data/annotations/task1/{train,val}/gt.csv`,
`UPAR-Challenge-2027/data/annotations/task2/val/{ids,queries}.csv` and the images under
`UPAR-Challenge-2027/data/{Market1501,PA100k,PETA}/`. On first use, `par/upar/data.py` packs the encoded images of a
split into `cache/{train,val}_bytes.npz`. Only the challenge data are used for training. No test image is used for
training or model selection.

## 2. Environment

Docker (recommended):

```bash
docker build -t upar-t2 .
docker run --gpus all --shm-size 16g -it \
  -v $PWD/UPAR-Challenge-2027:/workspace/UPAR-Challenge-2027 -v $PWD/runs:/workspace/runs upar-t2
```

or pip: `pip install -r requirements.txt` (tested with Python 3.12, torch 2.11.0+cu128, timm 1.0.30; the training
Docker image uses torch 2.9.1). The pretrained encoders (`vit_large_patch16_dinov3.lvd1689m`,
`vit_huge_plus_patch16_dinov3.lvd1689m`, `vit_large_patch14_clip_224.openai`) are downloaded by timm from the Hugging
Face Hub on first use.

## 3. Method

Three attribute models predict 40 attribute logits per gallery image. Each has a one-layer Pre-LN cross-attention
query head (40 learned queries, one per attribute, attend to the CLS and patch tokens). In every model the patch
embedding and the first 8 encoder blocks are frozen.

| File | Encoder | Input | Training |
|---|---|---|---|
| `student.pt` | DINOv3 ViT-L/16 | 176 × 96 | 20 epochs, 0.2 · label loss + 0.8 · distillation (BCE to the sigmoid of the mean teacher logits); half of the batches are mixup / CutMix images trained on the teacher targets only. Teachers: DINOv3 ViT-L/16 (176 × 96) and CLIP ViT-L/14 (168 × 84) with query heads, 3 epochs. Seed 2 |
| `hp.pt` | DINOv3 ViT-H+/16 | 224 × 112 | 3 epochs, BCE + 0.5 · group cross-entropy over the mutually exclusive attribute groups (age, hair length, lower-body type, upper-body colour, lower-body colour). Seed 1 |
| `m1.pt` | OpenAI CLIP ViT-L/14 + text score | 224 × 112 | same as `hp.pt`, then WiSE-FT 0.7 |

Shared settings: AdamW, weight decay 0.05, batch 64, lr 5e-5 for the final encoder LayerNorm and × 0.75 per block
going down, 10× for the head, cosine schedule with 0.5 epoch warm-up, gradient clipping at norm 1.0, drop path 0.1,
sampling with replacement with each dataset drawn ∝ N^0.5, EMA 0.9995 (the EMA weights are used), bf16, BCE with
DeepMAR weights and label smoothing 0.05, base augmentation (`par/upar/data.py`). All models train on the train + val
splits of PA100k, PETA and Market-1501.

Inference (`submission/run.py`, `rank_gallery`): each gallery image is resized to the input size of each model and
passes once through the three models (no flip). The logits are averaged. Each gallery image's logits are then
smoothed with those of its 50 nearest gallery neighbours, z' = 0.8 z + 0.2 · mean of the neighbours' z, where the
neighbours are found by cosine similarity of the L2-normalised mean patch token of the DINOv3 ViT-H+/16 model from the
same forward pass. With p = σ(z') clipped to [1e-4, 1 − 1e-4], a gallery image g is ranked for a binary query q by the
log-likelihood of the exact query, s(q, g) = Σ_k q_k log p_k + (1 − q_k) log(1 − p_k). No threshold, prior offset or
test-time augmentation is used.

## 4. Reproduce the leaderboard entry

### Rebuild the submitted zip from the released weights

```bash
hf download tannguyenbmt47/upar2027-track2 --local-dir weights
python submission/build_v13.py --student weights/student.pt --hp weights/hp.pt --m1 weights/m1.pt --out t2_v13.zip
python scripts/check_t2.py t2_v13.zip --dataset PETA
```

`build_v13.py` rebuilds `t2_stu_hpG_clipTGw70_rerank_gpu_v13.zip` file for file (same CRC-32 for all 330 files; only
the zip timestamps differ), except two NVIDIA C-RADIO helper files that v13 does not use (see the notes below).
`check_t2.py` imports the zip's `run.py`, calls `rank_gallery()` like the ingestion program and reports mADM on the
validation gallery of one dataset. The released models were trained on train + val, so this is a smoke test.

### Train v13

```bash
bash scripts/train_v13.sh        # -> runs/F_kdmixL_trainval, runs/F_HG, runs/F_TG_w70 and runs/t2_v13.zip
```

The script trains the two teachers of the student, the student, the DINOv3 ViT-H+/16 and CLIP ViT-L/14 models,
applies WiSE-FT to the CLIP model, exports bf16 weights and builds the zip. Every run fits on one 32 GB GPU. Results of
a new run differ slightly from the released weights because of GPU non-determinism.

## Notes

- `upar/models/radio_nv/cls_token.py` and `vit_patch_generator.py` (in `par/` and `submission/vendor/`) are NVIDIA files
  of the C-RADIO model and are not part of this repository. They are only needed for the C-RADIO encoder, not by v13,
  and `upar/models/radio.py` downloads them from `nvidia/C-RADIOv3-B` on Hugging Face on first use.
- License: Apache 2.0 for the code (see `LICENSE` and `THIRD_PARTY_NOTICES.md`); the weights keep the terms of the
  pretrained DINOv3 (DINOv3 License) and OpenAI CLIP (MIT License) weights.
