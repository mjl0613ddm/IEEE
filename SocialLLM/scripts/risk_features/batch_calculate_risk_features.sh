#!/bin/bash

###############################################################################
# 批量计算风险特征指标脚本
# 
# 该脚本用于批量处理指定模型的结果文件夹，计算5个风险特征指标
###############################################################################

# ============================================================================
# 配置参数（可根据需要修改）
# ============================================================================

# 模型名称列表（例如: gpt-4o-mini, qwen-plus, claude-3-haiku-20240307）
# 可以指定多个模型，用空格分隔
# 如果为空，则自动识别所有模型
MODEL_NAMES=()

# 是否强制覆盖已存在的结果文件
# true: 覆盖已存在的文件
# false: 跳过已存在的文件（默认）
FORCE_OVERWRITE=false

# 项目根目录路径
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 脚本路径
CALCULATE_SCRIPT="${BASE_PATH}/scripts/risk_features/calculate_risk_features.py"

# 结果目录
RESULTS_BASE_DIR="${BASE_PATH}/results"

# 日志文件
LOG_DIR="${BASE_PATH}/scripts/risk_features/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/batch_calculate_risk_features_${TIMESTAMP}.log"

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
    
    if [ ! -f "$CALCULATE_SCRIPT" ]; then
        error "风险特征计算脚本不存在: $CALCULATE_SCRIPT"
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
    
    log "=========================================="
    log "处理: ${model_name}/${result_dir_name}"
    log "=========================================="
    
    # 检查结果目录是否存在
    if [ ! -d "$result_dir_path" ]; then
        error "结果目录不存在: $result_dir_path"
        return 1
    fi
    
    # 检查results.json是否存在
    if [ ! -f "${result_dir_path}/results.json" ]; then
        error "结果文件不存在: ${result_dir_path}/results.json，跳过"
        return 1
    fi
    
    # 检查max_risk_timestep是否符合标准（>= 10）
    local max_risk_timestep=$(python3 -c "import json; data = json.load(open('${result_dir_path}/results.json')); print(data.get('max_risk_timestep', -1))" 2>/dev/null)
    if [ -z "$max_risk_timestep" ] || [ "$max_risk_timestep" = "-1" ] || [ "$max_risk_timestep" -lt 10 ]; then
        log "  跳过: max_risk_timestep ($max_risk_timestep) < 10，不符合筛选标准"
        return 0
    fi
    
    # 检查必要的数据文件
    if [ ! -d "${result_dir_path}/shapley" ]; then
        error "Shapley目录不存在: ${result_dir_path}/shapley，跳过"
        return 1
    fi
    
    if [ ! -d "${result_dir_path}/action_table" ]; then
        error "Action table目录不存在: ${result_dir_path}/action_table，跳过（需要先运行extract_action_table.py）"
        return 1
    fi
    
    # 检查结果文件是否已存在
    result_file="${result_dir_path}/risk_feature/risk_features.json"
    if [ -f "$result_file" ]; then
        if [ "$FORCE_OVERWRITE" = "true" ]; then
            log "  结果文件已存在，但配置为覆盖模式，将重新计算: $result_file"
        else
            log "  结果文件已存在，跳过: $result_file (如需覆盖，请设置 FORCE_OVERWRITE=true)"
            return 0
        fi
    fi
    
    # 构建命令
    local cmd="python3 \"$CALCULATE_SCRIPT\""
    cmd="${cmd} --result_dir \"$result_dir_path\""
    
    # 切换到项目根目录执行命令
    cd "$BASE_PATH" || {
        error "无法切换到项目根目录: $BASE_PATH"
        return 1
    }
    
    # 执行命令并捕获输出
    local start_time=$(date +%s)
    log "  执行命令: $cmd"
    
    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "  ✓ 风险特征计算完成 (耗时: ${duration}秒)"
        
        # 验证输出文件是否存在
        if [ -f "$result_file" ]; then
            log "    结果文件: $result_file"
            return 0
        else
            error "    警告: 计算完成但未找到输出文件: $result_file"
            return 1
        fi
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        error "  ✗ 风险特征计算失败 (耗时: ${duration}秒)"
        return 1
    fi
}

# 处理单个模型的所有结果目录
process_model() {
    local model_name="$1"
    local model_path="${RESULTS_BASE_DIR}/${model_name}"
    
    log "=========================================="
    log "处理模型: ${model_name}"
    log "=========================================="
    
    # 检查模型目录是否存在
    if [ ! -d "$model_path" ]; then
        error "模型目录不存在: $model_path"
        return 1
    fi
    
    # 获取所有结果目录（排除_rm结尾的）
    local result_dirs=()
    for item in "$model_path"/*; do
        if [ -d "$item" ]; then
            local folder_name=$(basename "$item")
            # 只处理不以_rm结尾的文件夹，且包含results.json
            if ! is_rm_folder "$folder_name" && [ -f "${item}/results.json" ]; then
                result_dirs+=("$folder_name")
            fi
        fi
    done
    
    if [ ${#result_dirs[@]} -eq 0 ]; then
        log "  没有找到符合条件的结果目录（跳过_rm结尾的文件夹）"
        return 0
    fi
    
    log "  找到 ${#result_dirs[@]} 个结果目录"
    
    local success_count=0
    local fail_count=0
    local skip_count=0
    
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

# 自动发现模型目录
find_model_directories() {
    local model_dirs=()
    
    for model_dir in "$RESULTS_BASE_DIR"/*; do
        if [ -d "$model_dir" ]; then
            model_name=$(basename "$model_dir")
            # 检查是否有非_rm的子文件夹
            has_valid_subfolder=false
            for subfolder in "$model_dir"/*; do
                if [ -d "$subfolder" ]; then
                    subfolder_name=$(basename "$subfolder")
                    if ! is_rm_folder "$subfolder_name"; then
                        has_valid_subfolder=true
                        break
                    fi
                fi
            done
            
            if [ "$has_valid_subfolder" = true ]; then
                model_dirs+=("$model_name")
            fi
        fi
    done
    
    printf '%s\n' "${model_dirs[@]}"
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force|--overwrite)
                FORCE_OVERWRITE=true
                shift
                ;;
            --no-force|--no-overwrite)
                FORCE_OVERWRITE=false
                shift
                ;;
            --help|-h)
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --force, --overwrite      强制覆盖已存在的结果文件"
                echo "  --no-force, --no-overwrite  跳过已存在的结果文件（默认）"
                echo "  --help, -h                显示此帮助信息"
                echo ""
                echo "注意: 也可以在脚本中直接修改 FORCE_OVERWRITE 变量来设置默认行为"
                exit 0
                ;;
            *)
                error "未知参数: $1"
                echo "使用 --help 查看帮助信息"
                exit 1
                ;;
        esac
    done
}

# ============================================================================
# 主程序
# ============================================================================

main() {
    # 解析命令行参数
    parse_args "$@"
    
    log "=========================================="
    log "批量计算风险特征指标"
    log "=========================================="
    log "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 如果未指定模型，自动发现
    if [ ${#MODEL_NAMES[@]} -eq 0 ]; then
        log "未指定模型列表，自动发现所有模型目录..."
        readarray -t MODEL_NAMES < <(find_model_directories)
        if [ ${#MODEL_NAMES[@]} -eq 0 ]; then
            error "未找到任何模型目录"
            exit 1
        fi
        log "发现 ${#MODEL_NAMES[@]} 个模型目录: ${MODEL_NAMES[*]}"
    fi
    
    log "配置:"
    log "  模型列表: ${MODEL_NAMES[*]}"
    log "  结果目录: $RESULTS_BASE_DIR"
    log "  覆盖模式: $FORCE_OVERWRITE (true=覆盖已存在文件, false=跳过已存在文件)"
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
