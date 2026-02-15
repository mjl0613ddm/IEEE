#!/bin/bash
# 绘制 world_metrics.csv 中的变量折线图
# 默认绘制 price_inflation_rate 和 risk_indicator_naive 两个指标

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PLOT_SCRIPT="$PROJECT_DIR/scripts/plot/plot_world_metrics.py"

# 默认变量
DEFAULT_VARIABLES=("price_inflation_rate" "risk_indicator_naive")

# 解析参数
DATA_FOLDER=""
VARIABLES=()
DATA_ROOT="$PROJECT_DIR/data"
OUTPUT_DIR=""
CSV_FILE=""
BASELINE_CSV=""
REAL_CSV=""
COMPARE_DIR=""
THRESHOLD=""
THRESHOLDS=()

# 显示帮助信息
show_help() {
    cat << EOF
用法: $0 [选项] [data_folder] [variables...]

默认行为：
  如果不指定变量，将绘制 price_inflation_rate 和 risk_indicator_naive

参数：
  data_folder          数据文件夹名称（相对于 data 目录），例如：gpt-4o-injection5p-10agents-50months
  
选项：
  -h, --help           显示此帮助信息
  -v, --variables      要绘制的变量（可以指定多个，用空格分隔）
                       如果不指定，默认绘制 price_inflation_rate 和 risk_indicator_naive
  -o, --output-dir     输出目录（默认：{data_folder}/plot）
  -c, --csv-file       直接指定 CSV 文件路径
  -b, --baseline-csv   Baseline CSV 文件路径（用于对比模式）
  -r, --real-csv       Real CSV 文件路径（用于对比模式）
  -d, --compare-dir    包含 baseline 和 real 子目录的父目录路径
  -t, --threshold     阈值（红色虚线），适用于所有变量
  --data-root          数据根目录路径（默认：$PROJECT_DIR/data）

示例：
  # 使用默认变量绘制指定数据文件夹
  $0 gpt-4o-injection5p-10agents-50months
  
  # 绘制指定变量
  $0 gpt-4o-injection5p-10agents-50months price interest_rate
  
  # 使用 --variables 选项
  $0 -v price price_inflation_rate gpt-4o-injection5p-10agents-50months
  
  # 指定输出目录
  $0 -o /path/to/output gpt-4o-injection5p-10agents-50months
  
  # 使用 CSV 文件
  $0 -c /path/to/world_metrics.csv
  
  # 对比模式
  $0 -d /path/to/shapley_dir

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--variables)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^- ]]; do
                VARIABLES+=("$1")
                shift
            done
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--csv-file)
            CSV_FILE="$2"
            shift 2
            ;;
        -b|--baseline-csv)
            BASELINE_CSV="$2"
            shift 2
            ;;
        -r|--real-csv)
            REAL_CSV="$2"
            shift 2
            ;;
        -d|--compare-dir)
            COMPARE_DIR="$2"
            shift 2
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        -*)
            echo "❌ 错误：未知选项 $1"
            show_help
            exit 1
            ;;
        *)
            if [[ -z "$DATA_FOLDER" ]] && [[ -z "$CSV_FILE" ]] && [[ -z "$COMPARE_DIR" ]]; then
                DATA_FOLDER="$1"
            else
                VARIABLES+=("$1")
            fi
            shift
            ;;
    esac
done

# 如果没有指定变量，使用默认变量
if [[ ${#VARIABLES[@]} -eq 0 ]]; then
    VARIABLES=("${DEFAULT_VARIABLES[@]}")
    echo "ℹ️  使用默认变量: ${VARIABLES[*]}"
fi

# 构建命令
CMD="python $PLOT_SCRIPT"

# 添加参数
if [[ -n "$DATA_FOLDER" ]]; then
    CMD="$CMD $DATA_FOLDER"
fi

CMD="$CMD ${VARIABLES[@]}"

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

if [[ -n "$CSV_FILE" ]]; then
    CMD="$CMD --csv-file $CSV_FILE"
fi

if [[ -n "$BASELINE_CSV" ]]; then
    CMD="$CMD --baseline-csv $BASELINE_CSV"
fi

if [[ -n "$REAL_CSV" ]]; then
    CMD="$CMD --real-csv $REAL_CSV"
fi

if [[ -n "$COMPARE_DIR" ]]; then
    CMD="$CMD --compare-dir $COMPARE_DIR"
fi

if [[ -n "$THRESHOLD" ]]; then
    CMD="$CMD --threshold $THRESHOLD"
fi

if [[ -n "$DATA_ROOT" ]]; then
    CMD="$CMD --data-root $DATA_ROOT"
fi

# 执行命令
echo "🚀 执行命令: $CMD"
echo ""
eval $CMD

