#!/bin/bash
# 批量计算 TwinMarket Shapley 值归因
# 可对单个 result_dir 运行，或对多个模型/seed 目录批量运行

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# ==================== 配置 ====================
# 单目录模式：直接指定结果目录
RESULT_DIR="${RESULT_DIR:-}"
# 或：结果根目录 + 模型/seed 子目录名（例如 results/gpt-4o-mini/gpt-4o-mini_46）
RESULTS_ROOT="${RESULTS_ROOT:-}"
MODEL_SEED="${MODEL_SEED:-}"   # 例如 gpt-4o-mini_46

N_SAMPLES="${N_SAMPLES:-1000}"
N_THREADS="${N_THREADS:-0}"   # 0 = 使用 CPU 核心数
START_DATE="${START_DATE:-}"
TARGET_DATE="${TARGET_DATE:-}"

# ==================== 解析 result_dir ====================
if [ -n "$RESULT_DIR" ]; then
    RESOLVED_RESULT_DIR="$(cd "$RESULT_DIR" 2>/dev/null && pwd)" || true
fi
if [ -z "$RESOLVED_RESULT_DIR" ] && [ -n "$RESULTS_ROOT" ] && [ -n "$MODEL_SEED" ]; then
    RESOLVED_RESULT_DIR="$(cd "$RESULTS_ROOT/$MODEL_SEED" 2>/dev/null && pwd)" || true
fi
if [ -z "$RESOLVED_RESULT_DIR" ]; then
    echo "请设置 RESULT_DIR 或 (RESULTS_ROOT + MODEL_SEED)"
    echo "示例: RESULT_DIR=/path/to/gpt-4o-mini_46 bash $0"
    echo "或:   RESULTS_ROOT=/path/to/results MODEL_SEED=gpt-4o-mini_46 bash $0"
    exit 1
fi

# ==================== 自动获取日期范围（可选）====================
if [ -z "$START_DATE" ] || [ -z "$TARGET_DATE" ]; then
    MARKET_CSV="$RESOLVED_RESULT_DIR/analysis/market_metrics.csv"
    if [ -f "$MARKET_CSV" ]; then
        echo "从 analysis/market_metrics.csv 推断日期范围..."
        OUT=$(python3 "$SCRIPT_DIR/find_max_risk_date.py" "$RESOLVED_RESULT_DIR" 2>/dev/null) || true
        if [ -n "$OUT" ]; then
            START_DATE="${START_DATE:-$(echo "$OUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('start_date',''))")}"
            TARGET_DATE="${TARGET_DATE:-$(echo "$OUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('target_date',''))")}"
        fi
    fi
fi
if [ -z "$START_DATE" ] || [ -z "$TARGET_DATE" ]; then
    echo "请设置 START_DATE 和 TARGET_DATE，或确保 result_dir 下存在 analysis/market_metrics.csv"
    exit 1
fi

echo "=========================================="
echo "TwinMarket Shapley 计算"
echo "=========================================="
echo "结果目录: $RESOLVED_RESULT_DIR"
echo "日期范围: $START_DATE -> $TARGET_DATE"
echo "采样数:   $N_SAMPLES"
echo "线程数:   $N_THREADS"
echo "=========================================="

python3 "$SCRIPT_DIR/calculate_shapley.py" \
    --result_dir "$RESOLVED_RESULT_DIR" \
    --start_date "$START_DATE" \
    --target_date "$TARGET_DATE" \
    --n_samples "$N_SAMPLES" \
    --n_threads "$N_THREADS" \
    --base_path "$PROJECT_ROOT"

echo "完成. 输出目录: $RESOLVED_RESULT_DIR/shapley/"
