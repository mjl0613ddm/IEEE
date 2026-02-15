#!/bin/bash
# 计算完全Shapley值和蒙特卡罗方法的误差
# 验证蒙特卡罗方法的近似准确性

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_SCRIPT="$PROJECT_DIR/scripts/accuracy_exp/compute_shapley_error.py"

# ==================== 配置参数（可修改） ====================
# 实验设置
NUM_AGENTS=4
EPISODE_LENGTH=5

# 数据路径（相对于项目根目录）
REAL_ACTIONS_JSON="data/llama-verify/actions_json/all_actions.json"
BASELINE_ACTIONS_JSON="data/baseline-verify-2/actions_json/all_actions.json"

# 归因指标
METRIC_NAME="risk_indicator_naive"
TARGET_TIMESTEP=""  # 如果为空，默认使用最后一个时间步（episode_length）
RISK_LAMBDA=0.94

# Baseline策略
BASELINE_TYPE="fixed"  # fixed, average, stable
BASELINE_WORK=1.0
BASELINE_CONSUMPTION=0.9
USE_PROBABILISTIC_BASELINE=false  # true 或 false

# 随机种子（用于多次运行取平均）
SEEDS=(42) 

# MC采样次数列表（用于测试不同采样次数的误差）
MC_SAMPLES_LIST=(10 100 1000 10000)

# 输出目录
OUTPUT_DIR="results/llama_shapley_error_analysis"

# 其他参数
INFLATION_THRESHOLD=0.0
USE_METRIC_DIRECTLY=false  # true 或 false
RISK_AGGREGATION="max"  # max 或 sum
EXCLUDE_BOTH_RISKS=false  # true 或 false
SKIP_EXACT=false  # true 或 false，如果为true则跳过完全Shapley计算（使用缓存的）
EXACT_SHAPLEY_FILE=  # 已有的完全Shapley值文件路径（.npy），如果指定则使用该文件而不是重新计算
N_JOBS=48  # 并行进程数，用于加速完全Shapley计算（1=串行，>1=并行，建议设置为CPU核心数）

# ==================== 命令行参数解析 ====================
show_help() {
    cat << EOF
用法: $0 [选项]

计算完全Shapley值和蒙特卡罗方法的误差，验证MC方法的近似准确性。

选项：
  -h, --help                    显示此帮助信息
  -n, --num-agents NUM           Agent数量（默认：$NUM_AGENTS）
  -e, --episode-length NUM       模拟周期数（默认：$EPISODE_LENGTH）
  -r, --real-actions PATH        Real actions JSON路径（默认：$REAL_ACTIONS_JSON）
  -b, --baseline-actions PATH    Baseline actions JSON路径（默认：$BASELINE_ACTIONS_JSON）
  -m, --metric-name NAME         归因指标名称（默认：$METRIC_NAME）
  -t, --target-timestep NUM      目标时间步（1-indexed，默认：最后一个时间步）
  -s, --seeds SEED1 SEED2 ...    随机种子列表（默认：${SEEDS[*]}）
  -c, --mc-samples LIST          MC采样次数列表（默认：${MC_SAMPLES_LIST[*]}）
  -o, --output-dir DIR           输出目录（默认：$OUTPUT_DIR）
  --skip-exact                   跳过完全Shapley计算（使用缓存的）
  --exact-shapley-file PATH       已有的完全Shapley值文件路径（.npy），如果指定则使用该文件而不是重新计算
  -j, --n-jobs NUM               并行进程数，用于加速完全Shapley计算（默认：1，>1时使用并行）

示例：
  # 使用默认配置运行
  $0
  
  # 修改agent数量和时间步数
  $0 -n 10 -e 10
  
  # 使用不同的数据路径
  $0 -r data/gpt-4o/actions_json/all_actions.json -b data/baseline-0.8/actions_json/all_actions.json
  
  # 修改MC采样次数列表
  $0 -c 100 500 1000 2000
  
  # 使用不同的种子
  $0 -s 100 101 102 103 104
  
  # 使用已有的完全Shapley值文件进行对比
  $0 --exact-shapley-file results/shapley_error_analysis/exact_shapley_values.npy

注意：
  - 完全Shapley值计算需要枚举所有2^(num_agents*episode_length)个子集，计算量较大
  - 对于5个agent、5个时间步，需要计算2^25=33,554,432个子集
  - 建议先使用--skip-exact跳过完全计算，测试MC方法

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -n|--num-agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        -e|--episode-length)
            EPISODE_LENGTH="$2"
            shift 2
            ;;
        -r|--real-actions)
            REAL_ACTIONS_JSON="$2"
            shift 2
            ;;
        -b|--baseline-actions)
            BASELINE_ACTIONS_JSON="$2"
            shift 2
            ;;
        -m|--metric-name)
            METRIC_NAME="$2"
            shift 2
            ;;
        -t|--target-timestep)
            TARGET_TIMESTEP="$2"
            shift 2
            ;;
        -s|--seeds)
            shift
            SEEDS=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^- ]]; do
                SEEDS+=("$1")
                shift
            done
            ;;
        -c|--mc-samples)
            shift
            MC_SAMPLES_LIST=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^- ]]; do
                MC_SAMPLES_LIST+=("$1")
                shift
            done
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-exact)
            SKIP_EXACT=true
            shift
            ;;
        --exact-shapley-file)
            EXACT_SHAPLEY_FILE="$2"
            shift 2
            ;;
        -j|--n-jobs)
            N_JOBS="$2"
            shift 2
            ;;
        -*)
            echo "❌ 错误：未知选项 $1"
            show_help
            exit 1
            ;;
        *)
            echo "❌ 错误：未知参数 $1"
            show_help
            exit 1
            ;;
    esac
done

# ==================== 参数验证 ====================
# 检查Python脚本是否存在
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "❌ 错误：找不到Python脚本: $PYTHON_SCRIPT"
    exit 1
fi

# 检查数据文件是否存在
REAL_ACTIONS_FULL="$PROJECT_DIR/$REAL_ACTIONS_JSON"
BASELINE_ACTIONS_FULL="$PROJECT_DIR/$BASELINE_ACTIONS_JSON"

if [[ ! -f "$REAL_ACTIONS_FULL" ]]; then
    echo "❌ 错误：找不到real actions文件: $REAL_ACTIONS_FULL"
    exit 1
fi

if [[ ! -f "$BASELINE_ACTIONS_FULL" ]]; then
    echo "❌ 错误：找不到baseline actions文件: $BASELINE_ACTIONS_FULL"
    exit 1
fi

# ==================== 构建命令 ====================
cd "$PROJECT_DIR"

CMD="python $PYTHON_SCRIPT"

# 基本参数
CMD="$CMD --num_agents $NUM_AGENTS"
CMD="$CMD --episode_length $EPISODE_LENGTH"
CMD="$CMD --real_actions_json $REAL_ACTIONS_JSON"
CMD="$CMD --baseline_actions_json $BASELINE_ACTIONS_JSON"
CMD="$CMD --metric_name $METRIC_NAME"
CMD="$CMD --risk_lambda $RISK_LAMBDA"

# Baseline参数
CMD="$CMD --baseline_type $BASELINE_TYPE"
if [[ "$BASELINE_TYPE" == "fixed" ]]; then
    CMD="$CMD --baseline_work $BASELINE_WORK"
    CMD="$CMD --baseline_consumption $BASELINE_CONSUMPTION"
fi

if [[ "$USE_PROBABILISTIC_BASELINE" == "true" ]]; then
    CMD="$CMD --use_probabilistic_baseline"
fi

# 时间步参数
if [[ -n "$TARGET_TIMESTEP" ]]; then
    CMD="$CMD --target_timestep $TARGET_TIMESTEP"
fi

# 随机种子
if [[ ${#SEEDS[@]} -gt 0 ]]; then
    CMD="$CMD --seeds ${SEEDS[*]}"
fi

# MC采样次数列表
if [[ ${#MC_SAMPLES_LIST[@]} -gt 0 ]]; then
    CMD="$CMD --mc_samples_list ${MC_SAMPLES_LIST[*]}"
fi

# 输出目录
CMD="$CMD --output_dir $OUTPUT_DIR"

# 其他参数
CMD="$CMD --inflation_threshold $INFLATION_THRESHOLD"
CMD="$CMD --risk_aggregation $RISK_AGGREGATION"

if [[ "$USE_METRIC_DIRECTLY" == "true" ]]; then
    CMD="$CMD --use_metric_directly"
fi

if [[ "$EXCLUDE_BOTH_RISKS" == "true" ]]; then
    CMD="$CMD --exclude_both_risks"
fi

if [[ "$SKIP_EXACT" == "true" ]]; then
    CMD="$CMD --skip_exact"
fi

if [[ -n "$EXACT_SHAPLEY_FILE" ]]; then
    CMD="$CMD --exact_shapley_file $EXACT_SHAPLEY_FILE"
fi

if [[ "$N_JOBS" -gt 1 ]]; then
    CMD="$CMD --n_jobs $N_JOBS"
fi

# ==================== 显示配置信息 ====================
echo "=========================================="
echo "Shapley Error Analysis Configuration"
echo "=========================================="
echo "实验设置:"
echo "  - Agent数量: $NUM_AGENTS"
echo "  - 时间步数: $EPISODE_LENGTH"
echo "  - 总Player数: $((NUM_AGENTS * EPISODE_LENGTH))"
echo "  - 完全Shapley需要计算的子集数: $((2 ** (NUM_AGENTS * EPISODE_LENGTH)))"
echo ""
echo "数据路径:"
echo "  - Real actions: $REAL_ACTIONS_JSON"
echo "  - Baseline actions: $BASELINE_ACTIONS_JSON"
echo ""
echo "归因设置:"
echo "  - 指标名称: $METRIC_NAME"
if [[ -n "$TARGET_TIMESTEP" ]]; then
    echo "  - 目标时间步: $TARGET_TIMESTEP"
else
    echo "  - 目标时间步: 最后一个时间步 ($EPISODE_LENGTH)"
fi
echo "  - Risk lambda: $RISK_LAMBDA"
echo ""
echo "Baseline策略:"
echo "  - 类型: $BASELINE_TYPE"
if [[ "$BASELINE_TYPE" == "fixed" ]]; then
    echo "  - Work: $BASELINE_WORK"
    echo "  - Consumption: $BASELINE_CONSUMPTION"
fi
echo "  - 概率性baseline: $USE_PROBABILISTIC_BASELINE"
echo ""
echo "实验配置:"
echo "  - 随机种子数量: ${#SEEDS[@]}"
echo "  - 种子列表: ${SEEDS[*]}"
echo "  - MC采样次数: ${MC_SAMPLES_LIST[*]}"
echo ""
echo "输出:"
echo "  - 输出目录: $OUTPUT_DIR"
if [[ "$SKIP_EXACT" == "true" ]]; then
    echo "  - ⚠️  跳过完全Shapley计算（使用缓存）"
fi
if [[ -n "$EXACT_SHAPLEY_FILE" ]]; then
    echo "  - 📁 使用已有的完全Shapley值文件: $EXACT_SHAPLEY_FILE"
fi
if [[ "$N_JOBS" -gt 1 ]]; then
    echo "  - ⚡ 并行计算: $N_JOBS 个进程"
fi
echo "=========================================="
echo ""

# ==================== 执行命令 ====================
echo "🚀 开始执行..."
echo "命令: $CMD"
echo ""
eval $CMD

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "✅ 执行成功！"
    echo "结果保存在: $OUTPUT_DIR"
    echo ""
    echo "输出文件："
    echo "  - exact_shapley_values.npy: 完全Shapley值"
    echo "  - mc_shapley_n*.npy: 不同采样次数的MC Shapley值"
    echo "  - error_analysis_results.json: 完整的误差分析结果"
    echo "  - error_analysis.csv: 误差分析表格（便于绘图）"
else
    echo ""
    echo "❌ 执行失败，退出码: $EXIT_CODE"
    exit $EXIT_CODE
fi

