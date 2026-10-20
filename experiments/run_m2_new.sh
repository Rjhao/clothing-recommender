#!/usr/bin/env bash
# M2 重训 — 改进 B 接入打分路径(α=0.1 有界混合)版本
# 用 abl_m2 原始超参,但 train.py 必须传 --use-multimodal 1 (适配 ablation ckpt 结构)。
#
# 产物:
#   checkpoints/abl_m2_new/best_model.pt + history.json
#   checkpoints/abl_m2_new/evaluation_results.json
set -u
PY="/c/Python311/python.exe"
DATA="data/amazon_fashion_real.json"
EPOCHS=20
TAG="abl_m2_new"
A=1; B=1; C=0
cd /d/clothing-recommender || { echo "cd failed"; exit 1; }
export PYTHONUTF8=1

echo "==================== [$TAG] A=$A B=$B C=$C (B α=0.1 fused mix) ===================="
echo "[$(date +%H:%M:%S)] training $TAG ..."
"$PY" -m recommender.train --data "$DATA" --tag "$TAG" --epochs "$EPOCHS" \
    --use-a "$A" --use-b "$B" --use-c "$C" --use-multimodal 1 \
    --weight-decay 1e-4 --lambda-reg 1e-4 --lambda-cl 0.1 --lambda-kg 0.1 \
    --batch-size 256 --lr 1e-3 --n-neg 4 --seed 42 \
    --output-dir checkpoints --save-every 10 \
    || { echo "!!!! [$TAG] TRAIN FAILED"; exit 1; }
echo "[$(date +%H:%M:%S)] evaluating $TAG ..."
"$PY" -m recommender.evaluate --model "checkpoints/$TAG/best_model.pt" --data "$DATA" \
    --use-a "$A" --use-b "$B" --use-c "$C" \
    || { echo "!!!! [$TAG] EVAL FAILED"; exit 1; }
echo "[$(date +%H:%M:%S)] [$TAG] done."