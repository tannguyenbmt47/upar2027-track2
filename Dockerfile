# Training / evaluation image (UPAR 2027 Track 2: leaderboard entry v13).
#   docker build -t upar-t2 .
#   docker run --gpus all --shm-size 16g -v $PWD/UPAR-Challenge-2027:/workspace/UPAR-Challenge-2027 \
#       -v $PWD/runs:/workspace/runs upar-t2 bash scripts/train_v13.sh
# CUDA 12.8 wheels run on Ampere, Ada, Hopper and Blackwell GPUs.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY par par
COPY submission submission
COPY scripts scripts
CMD ["bash"]
