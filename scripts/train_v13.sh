#!/bin/bash
# Train, export and package the Track 2 leaderboard entry v13 (mADM 0.6817 on the hidden test set).
# All models train on the train + val splits of PA100k, PETA and Market-1501. Each run fits on one 32 GB GPU.
#   1. the two teachers of the student: DINOv3 ViT-L/16 (176 x 96) and CLIP ViT-L/14 (168 x 84), query heads, 3 epochs
#   2. the student: DINOv3 ViT-L/16 (176 x 96), 20 epochs, 0.2 label loss + 0.8 distillation from the mean teacher
#      logits, half of the batches mixup / CutMix with teacher targets only
#   3. DINOv3 ViT-H+/16 and CLIP ViT-L/14 (text score) at 224 x 112, 3 epochs, BCE + 0.5 group cross-entropy;
#      WiSE-FT 0.7 for CLIP
#   4. bf16 export and the submission zip (runs/t2_v13.zip)
set -e
cd "$(dirname "$0")/../par"
R=../runs
BASE="--head query --head-kw n_layers=1 --freeze-blocks 8 --train-splits train val --no-best --workers ${WORKERS:-16}"

# 1. teachers of the student
python train.py --backbone vit_large_patch16_dinov3.lvd1689m --size 176 96 --epochs 3 --seed 1 $BASE --out $R/F_dinoL_trainval
python train.py --backbone vit_large_patch14_clip_224.openai --size 168 84 --epochs 3 --seed 1 $BASE --out $R/F_clipL_trainval

# 2. student
python train.py --backbone vit_large_patch16_dinov3.lvd1689m --size 176 96 --epochs 20 --seed 2 $BASE \
  --teacher $R/F_dinoL_trainval/last.pt,$R/F_clipL_trainval/last.pt --kd-alpha 0.8 --kd-mix 0.5 --out $R/F_kdmixL_trainval

# 3. DINOv3 ViT-H+/16 and CLIP ViT-L/14 members with group cross-entropy
GCE="--size 224 112 --head-kw n_layers=1 --epochs 3 --freeze-blocks 8 --train-splits train val --no-best --seed 1 \
  --group-ce 0.5 --workers ${WORKERS:-16}"
python train.py --backbone vit_huge_plus_patch16_dinov3.lvd1689m --head query $GCE --out $R/F_HG
python train.py --backbone vit_large_patch14_clip_224.openai --head query_text --text-init upar/models/clip_l14_text.pt \
  $GCE --out $R/F_TG
python make_wise_member.py $R/F_TG 0.7 $R/F_TG_w70

# 4. export and package
python export_member.py $R/F_kdmixL_trainval $R/F_kdmixL_trainval/student.pt $R/F_kdmixL_trainval/student_cfg.json
python export_member.py $R/F_HG $R/F_HG/hp.pt $R/F_HG/hp_cfg.json
python export_member.py $R/F_TG_w70 $R/F_TG_w70/m1.pt $R/F_TG_w70/m1_cfg.json
python ../submission/build_v13.py \
  --student $R/F_kdmixL_trainval/student.pt --student-cfg $R/F_kdmixL_trainval/student_cfg.json \
  --hp $R/F_HG/hp.pt --hp-cfg $R/F_HG/hp_cfg.json --m1 $R/F_TG_w70/m1.pt --m1-cfg $R/F_TG_w70/m1_cfg.json \
  --out $R/t2_v13.zip
