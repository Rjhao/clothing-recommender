#!/usr/bin/env bash
# 消融实验补跑 - view_sparse 数据, M0->M3 逐项消融, 20 epoch
# 串行跑(避免并发 OOM). 与 purchase 上 run_ablation_real.sh 完全同协议,
# 仅数据路径 + tag 前缀不同, 论文 §3.4.4 跨场景对比实验直接复用 purchase 实验脚本结构.
#
# 产物:
#   checkpoints/vs_abl_{m0,m1,m2,m3}/best_model.pt + history.json
#   checkpoints/vs_abl_{m0,m1,m2,m3}/evaluation_results.json   (evaluate.py 写死输出到这里)
#   experiments/results/ablation_viewsparse_summary.json         (由 aggregate_ablation_viewsparse.py 生成)
#
# 注意: evaluate.py 必须显式 --use-a/b/c 与训练一致, 否则 state_dict key 不匹配 load 失败.
set -u
PY="/c/Python311/python.exe"
DATA="data/amazon_fashion_view_sparse.json"
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

# 4 配置与 purchase 同协议
for cfg in "vs_abl_m0 0 0 0" "vs_abl_m1 1 0 0" "vs_abl_m2 1 1 0" "vs_abl_m3 1 1 1"; do
  run_one $cfg || echo "!!!! [$cfg] failed, continuing to next config"
done

echo "==================== ALL CONFIGS DONE ===================="
echo "[$(date +%H:%M:%S)] aggregating ..."
"$PY" experiments/aggregate_ablation_viewsparse.py || echo "!!!! aggregate failed"
echo "[$(date +%H:%M:%S)] all done."
