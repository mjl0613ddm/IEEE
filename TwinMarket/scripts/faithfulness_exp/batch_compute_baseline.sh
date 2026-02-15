#!/bin/bash

###############################################################################
# 批量计算Baseline方法脚本
# 
# 该脚本用于批量处理指定模型的结果文件夹，计算baseline方法的分数
# 支持的方法: random, loo, llm, mast
###############################################################################

# ============================================================================
# 配置参数（可根据需要修改）
# ============================================================================

# 模型名称列表（例如: gpt-4o-mini, qwen-plus, claude-3-haiku-20240307, deepseek-v3.2）
# 可以指定多个模型，用空格分隔
MODEL_NAMES=("deepseek-v3.2")

# Baseline方法列表（可以指定多个方法，用空格分隔）
# 可选: random, loo, llm, mast
BASELINE_METHODS=("random" "loo" "llm" "mast")

# 项目根目录路径
BASE_PATH="/mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket"

# 脚本路径
RANDOM_SCRIPT="${BASE_PATH}/scripts/faithfulness_exp/compute_random_baseline.py"
LOO_SCRIPT="${BASE_PATH}/scripts/faithfulness_exp/compute_loo_baseline.py"
LLM_SCRIPT="${BASE_PATH}/scripts/faithfulness_exp/compute_llm_baseline.py"
MAST_SCRIPT="${BASE_PATH}/scripts/faithfulness_exp/compute_mast_baseline.py"

# 结果目录
RESULTS_DIR="${BASE_PATH}/results"

# LLM相关配置（仅用于llm和mast方法）
LLM_CONFIG_FILE="${BASE_PATH}/config/api.yaml"  # LLM配置文件路径

# 日志文件
LOG_DIR="${BASE_PATH}/scripts/faithfulness_exp/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/batch_compute_baseline_${TIMESTAMP}.log"

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
    
    if [ ! -d "$RESULTS_DIR" ]; then
        error "结果目录不存在: $RESULTS_DIR"
        missing_files=$((missing_files + 1))
    fi
    
    if [ ! -f "$RANDOM_SCRIPT" ]; then
        error "Random baseline脚本不存在: $RANDOM_SCRIPT"
        missing_files=$((missing_files + 1))
    fi
    
    if [ ! -f "$LOO_SCRIPT" ]; then
        error "LOO baseline脚本不存在: $LOO_SCRIPT"
        missing_files=$((missing_files + 1))
    fi
    
    if [ ! -f "$LLM_SCRIPT" ]; then
        error "LLM baseline脚本不存在: $LLM_SCRIPT"
        missing_files=$((missing_files + 1))
    fi
    
    if [ ! -f "$MAST_SCRIPT" ]; then
        error "MAST baseline脚本不存在: $MAST_SCRIPT"
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

# 处理单个子文件夹的单个baseline方法
process_baseline_method() {
    local model_name="$1"
    local subfolder="$2"
    local method="$3"
    local subfolder_path="${RESULTS_DIR}/${model_name}/${subfolder}"
    local log_dir_name="${subfolder}"
    
    log "  - 处理 ${method} 方法..."
    
    # 检查子文件夹是否存在
    if [ ! -d "$subfolder_path" ]; then
        error "    子文件夹不存在: $subfolder_path"
        return 1
    fi
    
    # 检查shapley目录是否存在（需要从shapley_stats获取配置）
    if [ ! -d "${subfolder_path}/shapley" ]; then
        error "    Shapley目录不存在: ${subfolder_path}/shapley，跳过"
        return 1
    fi
    
    # 根据方法选择脚本和参数
    local script=""
    local cmd=""
    
    case "$method" in
        "random")
            script="$RANDOM_SCRIPT"
            cmd="python3 \"$script\" --log_dir \"$log_dir_name\" --results_dir \"$RESULTS_DIR\""
            ;;
        "loo")
            script="$LOO_SCRIPT"
            cmd="python3 \"$script\" --log_dir \"$log_dir_name\" --results_dir \"$RESULTS_DIR\""
            ;;
        "llm")
            script="$LLM_SCRIPT"
            cmd="python3 \"$script\" --log_dir \"$log_dir_name\" --results_dir \"$RESULTS_DIR\""
            # 检查配置文件是否存在
            if [ -n "$LLM_CONFIG_FILE" ]; then
                if [ -f "$LLM_CONFIG_FILE" ]; then
                    cmd="${cmd} --config \"$LLM_CONFIG_FILE\""
                else
                    error "    LLM配置文件不存在: $LLM_CONFIG_FILE，跳过"
                    return 1
                fi
            else
                error "    未指定LLM配置文件，跳过"
                return 1
            fi
            # 检查action_table是否存在（llm和mast需要）
            if [ ! -d "${subfolder_path}/action_table" ]; then
                error "    action_table目录不存在: ${subfolder_path}/action_table，跳过（需要先运行extract_action_features.py）"
                return 1
            fi
            ;;
        "mast")
            script="$MAST_SCRIPT"
            cmd="python3 \"$script\" --log_dir \"$log_dir_name\" --results_dir \"$RESULTS_DIR\""
            # 检查配置文件是否存在
            if [ -n "$LLM_CONFIG_FILE" ]; then
                if [ -f "$LLM_CONFIG_FILE" ]; then
                    cmd="${cmd} --config \"$LLM_CONFIG_FILE\""
                else
                    error "    LLM配置文件不存在: $LLM_CONFIG_FILE，跳过"
                    return 1
                fi
            else
                error "    未指定LLM配置文件，跳过"
                return 1
            fi
            # 检查action_table是否存在（llm和mast需要）
            if [ ! -d "${subfolder_path}/action_table" ]; then
                error "    action_table目录不存在: ${subfolder_path}/action_table，跳过（需要先运行extract_action_features.py）"
                return 1
            fi
            ;;
        *)
            error "    未知的方法: $method"
            return 1
            ;;
    esac
    
    # 切换到项目根目录执行命令
    cd "$BASE_PATH" || {
        error "    无法切换到项目根目录: $BASE_PATH"
        return 1
    }
    
    # 执行命令并捕获输出
    local start_time=$(date +%s)
    log "    执行命令: $cmd"
    
    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "    ✓ ${method} 方法完成 (耗时: ${duration}秒)"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        error "    ✗ ${method} 方法失败 (耗时: ${duration}秒)"
        return 1
    fi
}

# 处理单个子文件夹的所有baseline方法
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
    
    # 遍历所有baseline方法
    for method in "${BASELINE_METHODS[@]}"; do
        if process_baseline_method "$model_name" "$subfolder" "$method"; then
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
    log "批量计算Baseline方法"
    log "=========================================="
    log "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    log "配置:"
    log "  模型列表: ${MODEL_NAMES[*]}"
    log "  Baseline方法: ${BASELINE_METHODS[*]}"
    log "  结果目录: $RESULTS_DIR"
    log "  日志文件: $LOG_FILE"
    if [ -n "$LLM_CONFIG_FILE" ]; then
        log "  LLM配置文件: $LLM_CONFIG_FILE"
    fi
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
