#!/bin/bash

###############################################################################
# 批量保存风险数据脚本
# 
# 该脚本用于批量处理所有模型和seed的结果目录，保存风险数据为npy文件
###############################################################################

# ============================================================================
# 配置参数（可根据需要修改）
# ============================================================================

# 项目根目录路径
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 结果目录（基础路径）
RESULTS_BASE_DIR="${BASE_PATH}/results"

# 保存数据脚本路径
SAVE_SCRIPT="${BASE_PATH}/scripts/risk_features/save_risk_data.py"

# 日志文件
LOG_DIR="${BASE_PATH}/scripts/risk_features/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/batch_save_risk_data_${TIMESTAMP}.log"

# 是否强制覆盖已存在的数据文件
# true: 覆盖已存在的文件
# false: 跳过已存在的文件（默认）
FORCE_OVERWRITE=false

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

# 检查必要文件是否存在
check_dependencies() {
    local missing_files=0
    
    if [ ! -d "$RESULTS_BASE_DIR" ]; then
        error "结果目录不存在: $RESULTS_BASE_DIR"
        missing_files=$((missing_files + 1))
    fi
    
    if [ ! -f "$SAVE_SCRIPT" ]; then
        error "保存数据脚本不存在: $SAVE_SCRIPT"
        missing_files=$((missing_files + 1))
    fi
    
    if [ $missing_files -gt 0 ]; then
        error "缺少必要文件，退出"
        return 1
    fi
    
    return 0
}

# 保存单个结果目录的数据
save_result_data() {
    local result_dir="$1"
    local result_dir_name=$(basename "$result_dir")
    
    log "  处理: $result_dir_name"
    
    # 检查结果目录是否存在
    if [ ! -d "$result_dir" ]; then
        error "    结果目录不存在: $result_dir"
        return 1
    fi
    
    # 检查shapley目录是否存在
    if [ ! -d "${result_dir}/shapley" ]; then
        error "    Shapley目录不存在: ${result_dir}/shapley，跳过"
        return 1
    fi
    
    # 检查shapley文件是否存在
    if [ ! -f "${result_dir}/shapley/shapley_attribution_timeseries.csv" ]; then
        error "    Shapley CSV文件不存在，跳过"
        return 1
    fi
    
    # 检查是否已存在数据文件（如果不需要覆盖）
    if [ "$FORCE_OVERWRITE" != "true" ]; then
        if [ -f "${result_dir}/data/risk_timeseries.npy" ] && \
           [ -f "${result_dir}/data/shapley_values.npy" ] && \
           [ -f "${result_dir}/data/risk_evolution.npy" ]; then
            log "    数据文件已存在，跳过（使用 --force 强制覆盖）"
            return 0
        fi
    fi
    
    # 切换到项目根目录执行命令
    cd "$BASE_PATH" || {
        error "    无法切换到项目根目录: $BASE_PATH"
        return 1
    }
    
    # 执行命令
    local start_time=$(date +%s)
    local cmd="python3 \"$SAVE_SCRIPT\" --result_dir \"$result_dir\""
    
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
    
    # 获取所有结果目录（排除特殊目录）
    local result_dirs=()
    for item in "$model_dir"/*; do
        if [ -d "$item" ]; then
            local item_name=$(basename "$item")
            # 排除特殊目录，只处理以模型名开头的目录（seed目录）
            if [[ "$item_name" == ${model_name}_* ]] && [ -d "${item}/shapley" ]; then
                result_dirs+=("$item")
            fi
        fi
    done
    
    if [ ${#result_dirs[@]} -eq 0 ]; then
        log "  没有找到结果目录（需要包含shapley/目录）"
        return 0
    fi
    
    log "  找到 ${#result_dirs[@]} 个结果目录"
    
    # 遍历所有结果目录
    for result_dir in "${result_dirs[@]}"; do
        if save_result_data "$result_dir"; then
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
    log "批量保存风险数据"
    log "=========================================="
    log "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    log "配置:"
    log "  结果目录: $RESULTS_BASE_DIR"
    log "  保存脚本: $SAVE_SCRIPT"
    log "  日志文件: $LOG_FILE"
    log "  强制覆盖: $FORCE_OVERWRITE"
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
            # 排除特殊目录
            if [ "$model_name" != "risk_feature" ] && \
               [ "$model_name" != "faithfulness_exp" ] && \
               [ "$model_name" != "filter_results" ]; then
                model_dirs+=("$model_name")
            fi
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
    log "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
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

# 解析命令行参数
SPECIFIED_MODELS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_OVERWRITE=true
            shift
            ;;
        --models)
            shift
            # 收集指定的模型名称
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                SPECIFIED_MODELS+=("$1")
                shift
            done
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --force           强制覆盖已存在的数据文件"
            echo "  --models MODEL... 只处理指定的模型（用空格分隔）"
            echo "  -h, --help        显示帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                          # 处理所有模型"
            echo "  $0 --force                  # 强制覆盖已存在的文件"
            echo "  $0 --models gpt-4o-mini     # 只处理指定模型"
            exit 0
            ;;
        *)
            error "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 修改main函数以支持指定模型
if [ ${#SPECIFIED_MODELS[@]} -gt 0 ]; then
    # 如果指定了模型，只处理这些模型
    log "=========================================="
    log "批量保存风险数据（指定模型模式）"
    log "=========================================="
    log "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    log "指定模型: ${SPECIFIED_MODELS[*]}"
    log ""
    
    if ! check_dependencies; then
        error "依赖检查失败，退出"
        exit 1
    fi
    
    total_success=0
    total_fail=0
    
    for model_name in "${SPECIFIED_MODELS[@]}"; do
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
    log "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    log "总计: 成功 ${total_success} 个模型, 失败 ${total_fail} 个模型"
    log "详细日志请查看: $LOG_FILE"
    
    if [ $total_fail -eq 0 ]; then
        log "所有模型处理成功！"
        exit 0
    else
        log "部分模型处理失败，请查看日志文件"
        exit 1
    fi
else
    # 运行主程序（处理所有模型）
    main "$@"
fi
