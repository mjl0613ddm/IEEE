#!/bin/bash

###############################################################################
# 批量提取Action Table脚本
# 
# 该脚本用于批量处理指定模型的结果文件夹，提取action table
# 自动跳过_rm结尾的文件夹
###############################################################################

# ============================================================================
# 配置参数（可根据需要修改）
# ============================================================================

# 模型名称列表（例如: gpt-4o-mini, qwen-plus, claude-3-haiku-20240307）
# 可以指定多个模型，用空格分隔
# 如果为空，则自动识别所有模型
MODEL_NAMES=()

# 项目根目录路径
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 脚本路径
EXTRACT_SCRIPT="${BASE_PATH}/scripts/faithfulness_exp/extract_action_table.py"

# 结果目录
RESULTS_BASE_DIR="${BASE_PATH}/results"

# 日志文件
LOG_DIR="${BASE_PATH}/scripts/faithfulness_exp/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/batch_extract_action_table_${TIMESTAMP}.log"

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
    
    if [ ! -f "$EXTRACT_SCRIPT" ]; then
        error "Action table提取脚本不存在: $EXTRACT_SCRIPT"
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

# 处理单个结果目录
process_result_dir() {
    local model_name="$1"
    local result_dir_name="$2"
    local result_dir_path="${RESULTS_BASE_DIR}/${model_name}/${result_dir_name}"
    
    log "处理: ${model_name}/${result_dir_name}"
    
    # 检查结果目录是否存在
    if [ ! -d "$result_dir_path" ]; then
        error "结果目录不存在: $result_dir_path"
        return 1
    fi
    
    # 检查results.json和actions.json是否存在
    if [ ! -f "${result_dir_path}/results.json" ]; then
        error "结果文件不存在: ${result_dir_path}/results.json，跳过"
        return 1
    fi
    
    if [ ! -f "${result_dir_path}/actions.json" ]; then
        error "Actions文件不存在: ${result_dir_path}/actions.json，跳过"
        return 1
    fi
    
    # 检查是否已存在
    local output_file="${result_dir_path}/action_table/action_table.csv"
    if [ -f "$output_file" ]; then
        log "  Action table已存在，跳过: $output_file"
        return 0
    fi
    
    # 切换到项目根目录执行命令
    cd "$BASE_PATH" || {
        error "无法切换到项目根目录: $BASE_PATH"
        return 1
    }
    
    # 执行命令
    local start_time=$(date +%s)
    local cmd="python3 \"$EXTRACT_SCRIPT\" --result_dir \"$result_dir_path\" --skip-existing"
    
    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "  ✓ Action table提取完成 (耗时: ${duration}秒)"
        
        # 验证输出文件是否存在
        if [ -f "$output_file" ]; then
            log "    结果文件: $output_file"
            return 0
        else
            error "    警告: 提取完成但未找到输出文件: $output_file"
            return 1
        fi
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        error "  ✗ Action table提取失败 (耗时: ${duration}秒)"
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
    local skip_count=0
    
    # 获取所有结果目录（排除_rm结尾的）
    local result_dirs=()
    for item in "$model_dir"/*; do
        if [ -d "$item" ]; then
            local folder_name=$(basename "$item")
            # 只处理不以_rm结尾的文件夹，且包含results.json和actions.json
            if ! is_rm_folder "$folder_name" && [ -f "${item}/results.json" ] && [ -f "${item}/actions.json" ]; then
                result_dirs+=("$folder_name")
            fi
        fi
    done
    
    if [ ${#result_dirs[@]} -eq 0 ]; then
        log "  没有找到符合条件的结果目录（跳过_rm结尾的文件夹）"
        return 0
    fi
    
    log "  找到 ${#result_dirs[@]} 个结果目录"
    
    # 遍历所有结果目录
    for result_dir_name in "${result_dirs[@]}"; do
        if process_result_dir "$model_name" "$result_dir_name"; then
            success_count=$((success_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi
    done
    
    log "完成: ${model_name} (成功: ${success_count}, 失败: ${fail_count}, 跳过: ${skip_count})"
    
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
    log "批量提取Action Table"
    log "=========================================="
    log "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    log "配置:"
    if [ ${#MODEL_NAMES[@]} -eq 0 ]; then
        log "  模型列表: 自动识别所有模型"
    else
        log "  模型列表: ${MODEL_NAMES[*]}"
    fi
    log "  结果目录: $RESULTS_BASE_DIR"
    log "  日志文件: $LOG_FILE"
    log ""
    
    # 检查依赖
    if ! check_dependencies; then
        error "依赖检查失败，退出"
        exit 1
    fi
    
    # 获取模型列表
    local model_list=()
    if [ ${#MODEL_NAMES[@]} -eq 0 ]; then
        # 自动识别所有模型
        for item in "$RESULTS_BASE_DIR"/*; do
            if [ -d "$item" ]; then
                local model_name=$(basename "$item")
                model_list+=("$model_name")
            fi
        done
    else
        # 使用指定的模型列表
        model_list=("${MODEL_NAMES[@]}")
    fi
    
    if [ ${#model_list[@]} -eq 0 ]; then
        log "没有找到模型目录"
        exit 0
    fi
    
    log "找到 ${#model_list[@]} 个模型目录"
    log ""
    
    local total_success=0
    local total_fail=0
    
    # 遍历所有模型
    for model_name in "${model_list[@]}"; do
        if process_model "$model_name"; then
            total_success=$((total_success + 1))
        else
            total_fail=$((total_fail + 1))
        fi
    done
    
    log ""
    log "=========================================="
    log "批量提取完成"
    log "=========================================="
    log "总成功: ${total_success}"
    log "总失败: ${total_fail}"
    log "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    log "日志文件: $LOG_FILE"
    
    if [ $total_fail -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# 运行主程序
main
