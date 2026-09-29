#!/usr/bin/env bash
# 跑 TimesNet/PatchTST/Transformer 三个 backbone:
#   每个 backbone 一个进程 + 独立日志 run_<bb>.log + 独立结果 finally_<bb>.csv;
#   各进程退出时由 python 端自动汇总自己的 CSV (见 adapt_z_multimodal.py 尾部),
#   本脚本末尾只做跨 backbone 的合并报告。
#
# 用法: scripts/run_all_backbones.sh [data] [pred_lens] [seeds]
#   scripts/run_all_backbones.sh Agriculture 12,24 2023,2024
#   PARALLEL=0 scripts/run_all_backbones.sh   # 显存不够并发时改串行
set -uo pipefail
cd "$(dirname "$0")/.."

DATA="${1:-all}"
PRED_LENS="${2:-12,24,36,48}"
SEEDS="${3:-2023,2024,2025,2026,2027}"
PARALLEL="${PARALLEL:-1}"

BACKBONES=("SOFTS" "PatchTST" "TimesNet" "iTransformer")

pids=()
for bb in "${BACKBONES[@]}"; do
  log="run_qwen_${bb}.log"
  csv="finally_${bb}.csv"
  args=(--backbone "$bb" --data "$DATA" --pred_lens "$PRED_LENS" \
        --seeds "$SEEDS" --csv "$csv")
  echo "[run_all_backbones] ${args[*]} -> $log"
  if [ "$PARALLEL" = "1" ]; then
    nohup python adapt_z_multimodal.py "${args[@]}" > "$log" 2>&1 &
    pids+=($!)
  else
    python adapt_z_multimodal.py "${args[@]}" > "$log" 2>&1 \
      || echo "[run_all_backbones] $bb FAILED, see $log"
  fi
done

if [ "$PARALLEL" = "1" ]; then
  echo "[run_all_backbones] waiting for ${#pids[@]} runs: ${pids[*]}"
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" \
      || echo "[run_all_backbones] ${BACKBONES[$i]} FAILED, see run_${BACKBONES[$i]}.log"
  done
fi

# ── 跨 backbone 合并报告 (逐 backbone 的汇总已在各自 log 里) ──
csvs=()
for bb in "${BACKBONES[@]}"; do
  [ -f "finally_${bb}.csv" ] && csvs+=("finally_${bb}.csv")
done
if [ "${#csvs[@]}" -eq 0 ]; then
  echo "[run_all_backbones] no csv produced, skip merged report"
  exit 1
fi

MERGED="finally_all_backbones.csv"
awk 'FNR==1 && NR!=1{next} 1' "${csvs[@]}" > "$MERGED"
echo "[run_all_backbones] merged ${csvs[*]} -> $MERGED"

python - "$MERGED" <<'PY'
import sys
merged = sys.argv[1]
sys.argv = sys.argv[:1]      # 双保险: 防 adapt_z 模块级 argparse 劫持
from adapt_z_multimodal import report_by_backbone
report_by_backbone(merged, out_csv="iAdapter_avg_summary_by_backbone_ALL.csv")
PY