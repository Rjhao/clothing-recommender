#!/usr/bin/env bash
# 消融实验补跑 - 真实 purchase 数据, M0->M3 逐项消融, 20 epoch
# 串行跑(避免并发 OOM). 用 Python311 (装了 cu128 torch, 支持 RTX 5060 Blackwell).
#
# 产物:
#   checkpoints/abl_{m0,m1,m2,m3}/best_model.pt + history.json
#   checkpoints/abl_{m0,m1,m2,m3}/evaluation_results.json   (evaluate.py 写死输出到这里)
#   experiments/results/ablation_real_summary.json           (由 aggregate_ablation_real.py 生成)
#
# 注意: evaluate.py 没有 --output 参数, 且 --use-a/b/c 必须与训练一致,
#       否则 load_state_dict 会因 state_dict key 不匹配而失败
#       (hybrid_rec.py:134/143/151 用 if use_improvement_x 控制子模块创建).
set -u
PY="/c/Python311/python.exe"
DATA="data/amazon_fashion_real.json"
EPOCHS=20
cd /d/clothing-recommender || { echo "cd failed"; exit 1; }
export PYTHONUTF8=1

run_one() {
  local tag="$1" a="$2" b="$3" c="$4"
  echo "==================== [$tag] A=$a B=$b C=$c ===================="
  echo "[$(date +%H:%M:%S)] training $tag ..."
  "$PY" -m recommender.train --data "$DATA" --tag "$tag" --epochs "$EPOCHS" --use-a "$a" --use-b "$b" --use-c "$c" \
    || { echo "!!!! [$tag] TRAIN FAILED"; return 1; }
  echo "[$(date +%H:%M:%S)] evaluating $tag ..."
  "$PY" -m recommender.evaluate --model "checkpoints/$tag/best_model.pt" --data "$DATA" --use-a "$a" --use-b "$b" --use-c "$c" \
    || { echo "!!!! [$tag] EVAL FAILED"; return 1; }
  echo "[$(date +%H:%M:%S)] [$tag] done."
}

for cfg in "abl_m0 0 0 0" "abl_m1 1 0 0" "abl_m2 1 1 0" "abl_m3 1 1 1"; do
  run_one $cfg || echo "!!!! [$cfg] failed, continuing to next config"
done

echo "==================== ALL CONFIGS DONE ===================="
echo "[$(date +%H:%M:%S)] aggregating ..."
"$PY" experiments/aggregate_ablation_real.py || echo "!!!! aggregate failed"
echo "[$(date +%H:%M:%S)] all done."
