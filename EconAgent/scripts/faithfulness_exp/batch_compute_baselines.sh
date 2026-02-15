#!/bin/bash

###############################################################################
# 批量计算Faithfulness Baseline方法脚本
# 
# 该脚本用于批量处理指定模型，计算所有baseline方法的分数矩阵
# 包装了 batch_compute_baselines.py，提供更方便的配置和日志功能
###############################################################################

# ============================================================================
# 配置参数（可根据需要修改）
# ============================================================================

# 项目根目录路径
BASE_PATH="/mnt/shared-storage-user/meijilin/Economic_System_Attribution/ACL24-EconAgent"

# 要运行的方法列表（可以指定多个方法，用空格分隔）
# 可选: action_table, random, loo, llm, mast
# 如果不设置，将运行所有方法
METHODS=""  # 例如: METHODS="random loo" 或留空表示所有方法

# 要处理的模型路径列表（格式: "model/model_id"）
# 如果不设置，将自动查找所有不带_rm后缀的模型
MODEL_PATHS=""  # 例如: MODEL_PATHS="gpt/gpt_42 claude/claude_42" 或留空表示自动查找

# LLM和MAST方法的配置文件路径（可选）
# 如果不设置，脚本会自动查找 scripts/faithfulness_exp/config.yaml（统一配置路径）
LLM_CONFIG=""  # 例如: LLM_CONFIG="${BASE_PATH}/scripts/faithfulness_exp/config.yaml"

# 其他选项
SKIP_ACTION_TABLE=false  # 是否跳过action table提取步骤
NO_SKIP=false            # 是否不跳过已存在的文件（强制重新计算）

# 输出报告文件路径（可选）
OUTPUT_REPORT=""  # 例如: OUTPUT_REPORT="${BASE_PATH}/batch_baselines_report.json"

# ============================================================================
# 脚本路径和日志配置
# ============================================================================

# Python脚本路径
PYTHON_SCRIPT="${BASE_PATH}/scripts/faithfulness_exp/batch_compute_baselines.py"

# 日志文件目录
LOG_DIR="${BASE_PATH}/scripts/faithfulness_exp/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/batch_compute_baselines_${TIMESTAMP}.log"

# ============================================================================
# 函数定义
# ============================================================================

# 打印带时间戳的日志
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# 打印错误信息
error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

# 打印信息
info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*" | tee -a "$LOG_FILE"
}

# 检查必要文件是否存在
check_dependencies() {
    local missing_files=0
    
    if [ ! -d "$BASE_PATH" ]; then
        error "项目根目录不存在: $BASE_PATH"
        missing_files=$((missing_files + 1))
    fi
    
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        error "Python脚本不存在: $PYTHON_SCRIPT"
        missing_files=$((missing_files + 1))
    fi
    
    if [ $missing_files -gt 0 ]; then
        error "缺少必要文件，退出"
        return 1
    fi
    
    return 0
}

# 执行Python脚本
run_python_script() {
    # 切换到项目根目录
    cd "$BASE_PATH" || {
        error "无法切换到项目根目录: $BASE_PATH"
        return 1
    }
    
    # 构建命令参数（直接构建字符串，Python的argparse会自动处理）
    local cmd_args=()
    
    # 添加方法参数
    if [ -n "$METHODS" ]; then
        cmd_args+=(--methods)
        cmd_args+=($METHODS)  # 让bash自动分割
    fi
    
    # 添加配置文件参数（如果不指定，Python脚本会自动查找统一配置路径）
    if [ -n "$LLM_CONFIG" ]; then
        if [ ! -f "$LLM_CONFIG" ]; then
            error "指定的配置文件不存在: $LLM_CONFIG"
            error "将使用默认配置查找逻辑（scripts/faithfulness_exp/config.yaml）"
            # 不返回错误，让Python脚本自动查找
        else
            cmd_args+=(--config "$LLM_CONFIG")
        fi
    fi
    # 如果不指定LLM_CONFIG，Python脚本会自动从统一配置路径加载
    
    # 添加跳过action table选项
    if [ "$SKIP_ACTION_TABLE" = true ]; then
        cmd_args+=(--skip-action-table)
    fi
    
    # 添加不跳过选项
    if [ "$NO_SKIP" = true ]; then
        cmd_args+=(--no-skip)
    fi
    
    # 添加模型路径参数
    if [ -n "$MODEL_PATHS" ]; then
        cmd_args+=(--model-paths)
        cmd_args+=($MODEL_PATHS)  # 让bash自动分割
    fi
    
    # 添加输出报告参数
    if [ -n "$OUTPUT_REPORT" ]; then
        cmd_args+=(--output-report "$OUTPUT_REPORT")
    fi
    
    # 显示命令（用于日志）
    local cmd_display="python3 $PYTHON_SCRIPT"
    for arg in "${cmd_args[@]}"; do
        if [[ "$arg" =~ [[:space:]] ]]; then
            cmd_display="$cmd_display \"$arg\""
        else
            cmd_display="$cmd_display $arg"
        fi
    done
    log "执行命令: $cmd_display"
    log ""
    
    # 执行命令
    python3 "$PYTHON_SCRIPT" "${cmd_args[@]}"
    return $?
}

# ============================================================================
# 主程序
# ============================================================================

main() {
    log "=========================================="
    log "批量计算Faithfulness Baseline方法"
    log "=========================================="
    log "项目根目录: $BASE_PATH"
    log "Python脚本: $PYTHON_SCRIPT"
    log "日志文件: $LOG_FILE"
    log ""
    
    # 显示配置信息
    if [ -n "$METHODS" ]; then
        log "指定方法: $METHODS"
    else
        log "方法: 所有方法 (action_table, random, loo, llm, mast)"
    fi
    
    if [ -n "$MODEL_PATHS" ]; then
        log "指定模型: $MODEL_PATHS"
    else
        log "模型: 自动查找所有不带_rm后缀的模型"
    fi
    
    if [ -n "$LLM_CONFIG" ]; then
        log "LLM配置文件: $LLM_CONFIG"
    fi
    
    if [ "$SKIP_ACTION_TABLE" = true ]; then
        log "跳过action table提取"
    fi
    
    if [ "$NO_SKIP" = true ]; then
        log "不跳过已存在的文件（强制重新计算）"
    fi
    
    log ""
    
    # 检查依赖
    if ! check_dependencies; then
        error "依赖检查失败，退出"
        return 1
    fi
    
    # 执行Python脚本并捕获输出
    run_python_script 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    
    log ""
    if [ $exit_code -eq 0 ]; then
        log "=========================================="
        log "批量计算完成（成功）"
        log "=========================================="
    else
        error "=========================================="
        error "批量计算完成（有错误，退出码: $exit_code）"
        error "=========================================="
    fi
    
    log "详细日志已保存到: $LOG_FILE"
    
    return $exit_code
}

# ============================================================================
# 参数解析
# ============================================================================

# 解析命令行参数（如果有）
while [[ $# -gt 0 ]]; do
    case $1 in
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --model-paths)
            MODEL_PATHS="$2"
            shift 2
            ;;
        --config)
            LLM_CONFIG="$2"
            shift 2
            ;;
        --skip-action-table)
            SKIP_ACTION_TABLE=true
            shift
            ;;
        --no-skip)
            NO_SKIP=true
            shift
            ;;
        --output-report)
            OUTPUT_REPORT="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --methods METHOD1 METHOD2 ...    指定要运行的方法（用空格分隔）"
            echo "  --model-paths PATH1 PATH2 ...    指定要处理的模型路径（用空格分隔）"
            echo "  --config CONFIG_FILE             指定LLM配置文件路径"
            echo "  --skip-action-table              跳过action table提取步骤"
            echo "  --no-skip                        不跳过已存在的文件（强制重新计算）"
            echo "  --output-report REPORT_FILE      保存执行报告到JSON文件"
            echo "  --help                           显示此帮助信息"
            echo ""
            echo "或者直接在脚本中修改配置参数（推荐）"
            exit 0
            ;;
        *)
            error "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 运行主程序
main "$@"
exit $?
