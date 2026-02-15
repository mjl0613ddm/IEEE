#!/bin/bash

###############################################################################
# 批量计算Faithfulness实验脚本
# 
# 该脚本用于批量处理指定模型的结果文件夹，计算faithfulness指标
###############################################################################

# ============================================================================
# 配置参数（可根据需要修改）
# ============================================================================

# 模型名称列表（例如: gpt-4o-mini, qwen-plus, claude-3-haiku-20240307, deepseek-v3.2, llama-3.1-70b-instruct）
# 可以指定多个模型，用空格分隔
MODEL_NAMES=("gpt-4o-mini" "qwen-plus" "claude-3-haiku-20240307" "deepseek-v3.2" "llama-3.1-70b-instruct")

# 归因方法列表（可以指定多个方法，用空格分隔）
# 可选: shapley, random, llm, mast, loo
METHODS=("random" "llm" "mast" "loo" "shapley")

# 指标类型列表（可以指定多个指标类型，用空格分隔）
# 支持格式:
#   - deletion_top_n (n为任意正整数，如 deletion_top_3, deletion_top_15, deletion_top_20 等)
#   - insertion_top_n (n为任意正整数，如 insertion_top_3, insertion_top_15, insertion_top_20 等)
#   - deletion_low_n (n为任意正整数，如 deletion_low_10, deletion_low_20 等)
# 也支持旧的固定格式（向后兼容）: deletion_top_5, deletion_top_10, insertion_top_5, insertion_top_10, deletion_low_10
# 示例: METRIC_TYPES=("deletion_top_3" "deletion_top_5" "deletion_top_10" "deletion_top_20" "insertion_top_5" "deletion_low_10")
METRIC_TYPES=("deletion_top_1" "deletion_top_3" "deletion_top_5" "deletion_top_10")

# 项目根目录路径
BASE_PATH="/mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket"

# Faithfulness计算参数
MAX_ACTIONS=10                # 最大处理的action数量（None表示处理所有，但metric_type会自动确定）
VERBOSE=false                 # 是否输出详细信息

# 脚本路径
FAITHFULNESS_SCRIPT="${BASE_PATH}/scripts/faithfulness_exp/compute_faithfulness.py"

# 结果目录
RESULTS_DIR="${BASE_PATH}/results"

# 日志文件
LOG_DIR="${BASE_PATH}/scripts/faithfulness_exp/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/batch_compute_faithfulness_${TIMESTAMP}.log"

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
    
    if [ ! -d "$RESULTS_DIR" ]; then
        error "结果目录不存在: $RESULTS_DIR"
        missing_files=$((missing_files + 1))
    fi
    
    if [ ! -f "$FAITHFULNESS_SCRIPT" ]; then
        error "Faithfulness计算脚本不存在: $FAITHFULNESS_SCRIPT"
        missing_files=$((missing_files + 1))
    fi
    
    if [ $missing_files -gt 0 ]; then
        error "缺少必要文件，退出"
        return 1
    fi
    
    return 0
}

# 检查子文件夹名称是否以_rm结尾
is_rm_folder() {
    local folder_name="$1"
    if [[ "$folder_name" == *_rm ]]; then
        return 0  # 是_rm文件夹
    else
        return 1  # 不是_rm文件夹
    fi
}

# 处理单个子文件夹的单个方法的单个指标类型
process_faithfulness_method_metric() {
    local model_name="$1"
    local subfolder="$2"
    local method="$3"
    local metric_type="$4"
    local subfolder_path="${RESULTS_DIR}/${model_name}/${subfolder}"
    local log_dir_path="${model_name}/${subfolder}"
    
    log "    - 处理 ${method} 方法 / ${metric_type} 指标..."
    
    # 检查子文件夹是否存在
    if [ ! -d "$subfolder_path" ]; then
        error "      子文件夹不存在: $subfolder_path"
        return 1
    fi
    
    # 检查方法所需的数据文件是否存在（只在第一次检查，避免重复）
    # 这个检查应该在循环外部完成，但为了安全起见保留在这里
    case "$method" in
        "shapley")
            if [ ! -d "${subfolder_path}/shapley" ]; then
                error "      Shapley目录不存在: ${subfolder_path}/shapley，跳过"
                return 1
            fi
            ;;
        "random"|"loo"|"llm"|"mast")
            if [ ! -d "${subfolder_path}/faithfulness_exp/${method}" ]; then
                error "      ${method}方法的数据目录不存在: ${subfolder_path}/faithfulness_exp/${method}，跳过（需要先运行compute_*_baseline.py）"
                return 1
            fi
            ;;
        *)
            error "      未知的方法: $method"
            return 1
            ;;
    esac
    
    # 构建命令
    local cmd="python3 \"$FAITHFULNESS_SCRIPT\""
    cmd="${cmd} --log_dir \"results/${log_dir_path}\""
    cmd="${cmd} --method \"$method\""
    cmd="${cmd} --metric_type \"$metric_type\""
    
    # 注意：max_actions参数会根据metric_type自动确定（如deletion_top_5对应5，deletion_top_20对应20等）
    # metric_type会自动覆盖max_actions的值，所以这里可以省略，或者保留作为备用
    if [ -n "$MAX_ACTIONS" ] && [ "$MAX_ACTIONS" != "None" ]; then
        cmd="${cmd} --max_actions $MAX_ACTIONS"
    fi
    
    if [ "$VERBOSE" = true ]; then
        cmd="${cmd} --verbose"
    fi
    
    # 切换到项目根目录执行命令
    cd "$BASE_PATH" || {
        error "      无法切换到项目根目录: $BASE_PATH"
        return 1
    }
    
    # 执行命令并捕获输出
    local start_time=$(date +%s)
    log "      执行命令: $cmd"
    
    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "      ✓ ${method}/${metric_type} 完成 (耗时: ${duration}秒)"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        error "      ✗ ${method}/${metric_type} 失败 (耗时: ${duration}秒)"
        return 1
    fi
}

# 处理单个子文件夹的单个方法（所有指标类型）
process_faithfulness_method() {
    local model_name="$1"
    local subfolder="$2"
    local method="$3"
    local subfolder_path="${RESULTS_DIR}/${model_name}/${subfolder}"
    
    log "  - 处理 ${method} 方法..."
    
    # 检查子文件夹是否存在
    if [ ! -d "$subfolder_path" ]; then
        error "    子文件夹不存在: $subfolder_path"
        return 1
    fi
    
    # 检查方法所需的数据文件是否存在（只检查一次）
    case "$method" in
        "shapley")
            if [ ! -d "${subfolder_path}/shapley" ]; then
                error "    Shapley目录不存在: ${subfolder_path}/shapley，跳过该方法"
                return 1
            fi
            ;;
        "random"|"loo"|"llm"|"mast")
            if [ ! -d "${subfolder_path}/faithfulness_exp/${method}" ]; then
                error "    ${method}方法的数据目录不存在: ${subfolder_path}/faithfulness_exp/${method}，跳过该方法（需要先运行compute_*_baseline.py）"
                return 1
            fi
            ;;
        *)
            error "    未知的方法: $method"
            return 1
            ;;
    esac
    
    local method_success_count=0
    local method_fail_count=0
    
    # 遍历所有指标类型
    for metric_type in "${METRIC_TYPES[@]}"; do
        if process_faithfulness_method_metric "$model_name" "$subfolder" "$method" "$metric_type"; then
            method_success_count=$((method_success_count + 1))
        else
            method_fail_count=$((method_fail_count + 1))
        fi
    done
    
    log "    ${method} 方法完成 (成功: ${method_success_count}/${#METRIC_TYPES[@]}, 失败: ${method_fail_count})"
    
    if [ $method_fail_count -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

# 处理单个子文件夹的所有方法
process_subfolder() {
    local model_name="$1"
    local subfolder="$2"
    local subfolder_path="${RESULTS_DIR}/${model_name}/${subfolder}"
    
    log "=========================================="
    log "处理: ${model_name}/${subfolder}"
    log "=========================================="
    
    # 检查子文件夹是否存在
    if [ ! -d "$subfolder_path" ]; then
        error "子文件夹不存在: $subfolder_path"
        return 1
    fi
    
    local success_count=0
    local fail_count=0
    
    # 遍历所有方法
    for method in "${METHODS[@]}"; do
        if process_faithfulness_method "$model_name" "$subfolder" "$method"; then
            success_count=$((success_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi
    done
    
    log "完成: ${model_name}/${subfolder} (成功: ${success_count}, 失败: ${fail_count})"
    
    if [ $fail_count -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

# 处理单个模型的所有子文件夹
process_model() {
    local model_name="$1"
    local model_path="${RESULTS_DIR}/${model_name}"
    
    log "=========================================="
    log "处理模型: ${model_name}"
    log "=========================================="
    
    # 检查模型目录是否存在
    if [ ! -d "$model_path" ]; then
        error "模型目录不存在: $model_path"
        return 1
    fi
    
    # 获取所有子文件夹（排除_rm结尾的）
    local subfolders=()
    for item in "$model_path"/*; do
        if [ -d "$item" ]; then
            local folder_name=$(basename "$item")
            # 只处理不以_rm结尾的文件夹
            if ! is_rm_folder "$folder_name"; then
                subfolders+=("$folder_name")
            fi
        fi
    done
    
    if [ ${#subfolders[@]} -eq 0 ]; then
        log "  没有找到符合条件的子文件夹（跳过_rm结尾的文件夹）"
        return 0
    fi
    
    log "  找到 ${#subfolders[@]} 个子文件夹"
    
    local success_count=0
    local fail_count=0
    
    # 遍历所有子文件夹
    for subfolder in "${subfolders[@]}"; do
        if process_subfolder "$model_name" "$subfolder"; then
            success_count=$((success_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi
    done
    
    log "模型 ${model_name} 处理完成 (成功: ${success_count}, 失败: ${fail_count})"
    
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
    log "批量计算Faithfulness实验"
    log "=========================================="
    log "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    log "配置:"
    log "  模型列表: ${MODEL_NAMES[*]}"
    log "  方法列表: ${METHODS[*]}"
    log "  指标类型列表: ${METRIC_TYPES[*]}"
    log "  最大action数: ${MAX_ACTIONS:-根据metric_type自动确定}"
    log "  详细输出: $VERBOSE"
    log "  结果目录: $RESULTS_DIR"
    log "  日志文件: $LOG_FILE"
    log ""
    
    # 检查依赖
    if ! check_dependencies; then
        error "依赖检查失败，退出"
        exit 1
    fi
    
    local total_success=0
    local total_fail=0
    
    # 遍历所有模型
    for model_name in "${MODEL_NAMES[@]}"; do
        if process_model "$model_name"; then
            total_success=$((total_success + 1))
        else
            total_fail=$((total_fail + 1))
        fi
    done
    
    log ""
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

# 运行主程序
main "$@"
