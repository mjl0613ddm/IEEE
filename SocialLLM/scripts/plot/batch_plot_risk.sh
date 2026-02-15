#!/bin/bash

###############################################################################
# 批量绘制风险折线图脚本
# 
# 该脚本用于批量绘制results目录下所有模型结果的风险折线图
###############################################################################

# ============================================================================
# 配置参数（可根据需要修改）
# ============================================================================

# 项目根目录路径
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 结果目录（基础路径）
RESULTS_BASE_DIR="${BASE_PATH}/results"

# 绘图脚本路径
PLOT_SCRIPT="${BASE_PATH}/scripts/plot/plot_risk.py"

# 日志文件
LOG_DIR="${BASE_PATH}/scripts/plot/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/batch_plot_risk_${TIMESTAMP}.log"

# ============================================================================
# 函数定义
# ============================================================================

# 打印带时间戳的日志
log() {
    echo "[$(date '+%Y-%m-%d %H:%M%S')] $*" | tee -a "$LOG_FILE"
}

# 打印错误信息
error() {
    echo "[$(date '+%Y-%m-%d %H:%M%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

# 检查必要文件是否存在
check_dependencies() {
    local missing_files=0
    
    if [ ! -d "$RESULTS_BASE_DIR" ]; then
        error "结果目录不存在: $RESULTS_BASE_DIR"
        missing_files=$((missing_files + 1))
    fi
    
    if [ ! -f "$PLOT_SCRIPT" ]; then
        error "绘图脚本不存在: $PLOT_SCRIPT"
        missing_files=$((missing_files + 1))
    fi
    
    if [ $missing_files -gt 0 ]; then
        error "缺少必要文件，退出"
        return 1
    fi
    
    return 0
}

# 绘制单个结果目录的图表
plot_result() {
    local result_dir="$1"
    local result_dir_name=$(basename "$result_dir")
    
    log "  绘制: $result_dir_name"
    
    # 检查结果目录是否存在
    if [ ! -d "$result_dir" ]; then
        error "    结果目录不存在: $result_dir"
        return 1
    fi
    
    # 检查results.json是否存在
    if [ ! -f "${result_dir}/results.json" ]; then
        error "    结果文件不存在: ${result_dir}/results.json，跳过"
        return 1
    fi
    
    # 切换到项目根目录执行命令
    cd "$BASE_PATH" || {
        error "    无法切换到项目根目录: $BASE_PATH"
        return 1
    }
    
    # 执行命令
    local start_time=$(date +%s)
    local cmd="python3 \"$PLOT_SCRIPT\" \"$result_dir\""
    
    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "    ✓ $result_dir_name 完成 (耗时: ${duration}秒)"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        error "    ✗ $result_dir_name 失败 (耗时: ${duration}秒)"
        return 1
    fi
}

# 处理单个模型的所有结果目录
process_model() {
    local model_name="$1"
    local model_dir="${RESULTS_BASE_DIR}/${model_name}"
    
    log "=========================================="
    log "处理模型: ${model_name}"
    log "=========================================="
    
    # 检查模型目录是否存在
    if [ ! -d "$model_dir" ]; then
        error "模型目录不存在: $model_dir"
        return 1
    fi
    
    local success_count=0
    local fail_count=0
    
    # 获取所有结果目录（排除plot目录）
    local result_dirs=()
    for item in "$model_dir"/*; do
        if [ -d "$item" ]; then
            local item_name=$(basename "$item")
            # 排除plot目录和其他特殊目录
            if [ "$item_name" != "plot" ] && [ -f "${item}/results.json" ]; then
                result_dirs+=("$item")
            fi
        fi
    done
    
    if [ ${#result_dirs[@]} -eq 0 ]; then
        log "  没有找到结果目录（需要包含results.json）"
        return 0
    fi
    
    log "  找到 ${#result_dirs[@]} 个结果目录"
    
    # 遍历所有结果目录
    for result_dir in "${result_dirs[@]}"; do
        if plot_result "$result_dir"; then
            success_count=$((success_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi
    done
    
    log "完成: ${model_name} (成功: ${success_count}, 失败: ${fail_count})"
    
    if [ $fail_count -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

# ============================================================================
# 主程序
# ============================================================================

main() {
    log "=========================================="
    log "批量绘制风险折线图"
    log "=========================================="
    log "开始时间: $(date '+%Y-%m-%d %H:%M%S')"
    log "配置:"
    log "  结果目录: $RESULTS_BASE_DIR"
    log "  绘图脚本: $PLOT_SCRIPT"
    log "  日志文件: $LOG_FILE"
    log ""
    
    # 检查依赖
    if ! check_dependencies; then
        error "依赖检查失败，退出"
        exit 1
    fi
    
    # 获取所有模型目录
    local model_dirs=()
    for item in "$RESULTS_BASE_DIR"/*; do
        if [ -d "$item" ]; then
            local model_name=$(basename "$item")
            model_dirs+=("$model_name")
        fi
    done
    
    if [ ${#model_dirs[@]} -eq 0 ]; then
        log "没有找到模型目录"
        exit 0
    fi
    
    log "找到 ${#model_dirs[@]} 个模型目录"
    log ""
    
    local total_success=0
    local total_fail=0
    
    # 遍历所有模型目录
    for model_name in "${model_dirs[@]}"; do
        if process_model "$model_name"; then
            total_success=$((total_success + 1))
        else
            total_fail=$((total_fail + 1))
        fi
        log ""
    done
    
    log "=========================================="
    log "批量处理完成"
    log "=========================================="
    log "结束时间: $(date '+%Y-%m-%d %H:%M%S')"
    log "总计: 成功 ${total_success} 个模型, 失败 ${total_fail} 个模型"
    log "详细日志请查看: $LOG_FILE"
    
    if [ $total_fail -eq 0 ]; then
        log "所有模型处理成功！"
        exit 0
    else
        log "部分模型处理失败，请查看日志文件"
        exit 1
    fi
}

# 运行主程序
main "$@"
