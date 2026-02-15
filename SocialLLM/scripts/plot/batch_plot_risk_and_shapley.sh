#!/bin/bash
# -*- coding: utf-8 -*-
#
# 批量绘制SocialLLM的风险曲线和累积Shapley值图
#
# 用法:
#     bash scripts/plot/batch_plot_risk_and_shapley.sh [--models MODEL1 MODEL2 ...]
#
# 示例:
#     bash scripts/plot/batch_plot_risk_and_shapley.sh
#     bash scripts/plot/batch_plot_risk_and_shapley.sh --models gpt-4o-mini llama-3.1-8b-instruct
#

set -e  # 遇到错误立即退出

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 配置
RESULTS_BASE_DIR="$PROJECT_ROOT/results"
PLOT_SCRIPT="$SCRIPT_DIR/plot_risk_and_shapley.py"
LOG_DIR="$SCRIPT_DIR/logs"

# 是否强制覆盖已存在的图表文件
# true: 覆盖已存在的文件
# false: 跳过已存在的文件（默认）
FORCE_OVERWRITE=false

# 创建日志目录
mkdir -p "$LOG_DIR"

# 生成日志文件名（包含时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/batch_plot_risk_and_shapley_${TIMESTAMP}.log"

# 解析命令行参数
MODEL_NAMES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODEL_NAMES+=("$1")
                shift
            done
            ;;
        --force|--overwrite)
            FORCE_OVERWRITE=true
            shift
            ;;
        --no-force|--no-overwrite)
            FORCE_OVERWRITE=false
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项] [--models MODEL1 MODEL2 ...]"
            echo ""
            echo "选项:"
            echo "  --models MODEL1 MODEL2 ...  指定要处理的模型列表（可选）"
            echo "  --force, --overwrite        强制覆盖已存在的图表文件"
            echo "  --no-force, --no-overwrite  跳过已存在的图表文件（默认）"
            echo "  --help, -h                  显示此帮助信息"
            echo ""
            echo "注意: 也可以在脚本中直接修改 FORCE_OVERWRITE 变量来设置默认行为"
            exit 0
            ;;
        *)
            echo "未知参数: $1" >&2
            echo "用法: $0 [选项] [--models MODEL1 MODEL2 ...]" >&2
            echo "使用 --help 查看帮助信息" >&2
            exit 1
            ;;
    esac
done

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# 开始处理
log "=========================================="
log "批量绘制风险曲线和累积Shapley值图"
log "=========================================="
log "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 如果未指定模型列表，自动发现所有模型目录
if [ ${#MODEL_NAMES[@]} -eq 0 ]; then
    log "未指定模型列表，自动发现所有模型目录..."
    for dir in "$RESULTS_BASE_DIR"/*; do
        if [ -d "$dir" ]; then
            dirname=$(basename "$dir")
            # 排除特殊目录
            if [[ "$dirname" != "faithfulness_exp" && "$dirname" != "risk_feature" ]]; then
                MODEL_NAMES+=("$dirname")
            fi
        fi
    done
    log "发现 ${#MODEL_NAMES[@]} 个模型目录: ${MODEL_NAMES[*]}"
fi

log ""
log "配置:"
log "  模型列表: ${MODEL_NAMES[*]}"
log "  结果目录: $RESULTS_BASE_DIR"
log "  覆盖模式: $FORCE_OVERWRITE (true=覆盖已存在文件, false=跳过已存在文件)"
log "  日志文件: $LOG_FILE"
log ""

# 统计信息
TOTAL_PROCESSED=0
TOTAL_SUCCESS=0
TOTAL_FAILED=0
MODEL_SUCCESS=0
MODEL_FAILED=0

# 遍历每个模型
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    MODEL_DIR="$RESULTS_BASE_DIR/$MODEL_NAME"
    
    if [ ! -d "$MODEL_DIR" ]; then
        log "警告: 模型目录不存在，跳过: $MODEL_DIR"
        MODEL_FAILED=$((MODEL_FAILED + 1))
        continue
    fi
    
    log "=========================================="
    log "处理模型: $MODEL_NAME"
    log "=========================================="
    
    # 获取所有子文件夹（排除*_rm后缀的）
    SEED_COUNT=0
    PROCESSED_COUNT=0
    SUCCESS_COUNT=0
    FAILED_COUNT=0
    
    for SEED_DIR in "$MODEL_DIR"/*; do
        if [ ! -d "$SEED_DIR" ]; then
            continue
        fi
        
        SEED_NAME=$(basename "$SEED_DIR")
        
        # 跳过*_rm后缀的文件夹
        if [[ "$SEED_NAME" == *_rm ]]; then
            continue
        fi
        
        SEED_COUNT=$((SEED_COUNT + 1))
        
        # 检查必要文件是否存在
        RESULTS_JSON="$SEED_DIR/results.json"
        SHAPLEY_CSV="$SEED_DIR/shapley/shapley_attribution_timeseries.csv"
        
        if [ ! -f "$RESULTS_JSON" ]; then
            log "  跳过 $SEED_NAME: results.json不存在"
            continue
        fi
        
        if [ ! -f "$SHAPLEY_CSV" ]; then
            log "  跳过 $SEED_NAME: shapley/shapley_attribution_timeseries.csv不存在"
            continue
        fi
        
        # 检查输出文件是否已存在
        OUTPUT_FILE="$SEED_DIR/plot/risk_and_cumulative_shapley.png"
        if [ -f "$OUTPUT_FILE" ]; then
            if [ "$FORCE_OVERWRITE" = "true" ]; then
                log "  图表已存在，但配置为覆盖模式，将重新绘制: $SEED_NAME"
            else
                log "  跳过 $SEED_NAME: 图表已存在 (如需覆盖，请使用 --force 选项)"
                continue
            fi
        fi
        
        PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
        TOTAL_PROCESSED=$((TOTAL_PROCESSED + 1))
        
        log "  处理: $SEED_NAME"
        
        # 运行绘图脚本
        if python3 "$PLOT_SCRIPT" --result_dir "$SEED_DIR" >> "$LOG_FILE" 2>&1; then
            log "    ✓ 成功"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
        else
            log "    ✗ 失败"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            TOTAL_FAILED=$((TOTAL_FAILED + 1))
        fi
    done
    
    log "  找到 $SEED_COUNT 个子文件夹"
    log "  处理了 $PROCESSED_COUNT 个，成功 $SUCCESS_COUNT 个，失败 $FAILED_COUNT 个"
    
    if [ $FAILED_COUNT -eq 0 ]; then
        MODEL_SUCCESS=$((MODEL_SUCCESS + 1))
    else
        MODEL_FAILED=$((MODEL_FAILED + 1))
    fi
    
    log ""
done

# 输出统计信息
log ""
log "=========================================="
log "批量处理完成"
log "=========================================="
log "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
log "总计: 成功 $TOTAL_SUCCESS 个, 失败 $TOTAL_FAILED 个, 总处理 $TOTAL_PROCESSED 个"
log "模型: 成功 $MODEL_SUCCESS 个, 失败 $MODEL_FAILED 个, 总计 ${#MODEL_NAMES[@]} 个"
log "详细日志请查看: $LOG_FILE"

if [ $TOTAL_FAILED -eq 0 ]; then
    log "所有结果处理成功！"
    exit 0
else
    log "有 $TOTAL_FAILED 个结果处理失败，请查看日志"
    exit 1
fi
