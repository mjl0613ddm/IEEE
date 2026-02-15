#!/bin/bash

###############################################################################
# 批量计算Baseline方法脚本
# 
# 该脚本用于批量处理指定模型的结果文件夹，计算baseline方法的分数
# 支持的方法: random, loo, llm, mast
# 自动跳过_rm结尾的文件夹
###############################################################################

# ============================================================================
# 配置参数（可根据需要修改）
# ============================================================================

# 模型名称列表（例如: gpt-4o-mini, qwen-plus, claude-3-haiku-20240307）
# 可以指定多个模型，用空格分隔
# 如果为空，则自动识别所有模型
MODEL_NAMES=()

# Baseline方法列表（可以指定多个方法，用空格分隔）
# 可选: random, loo, llm, mast
# 如果为空，则运行所有方法
BASELINE_METHODS=("loo")

# 是否强制覆盖已存在的结果文件
# true: 覆盖已存在的文件
# false: 跳过已存在的文件（默认）
FORCE_OVERWRITE=false

# 项目根目录路径
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 脚本路径
EXTRACT_SCRIPT="${BASE_PATH}/scripts/faithfulness_exp/extract_action_table.py"
RANDOM_SCRIPT="${BASE_PATH}/scripts/faithfulness_exp/compute_random_baseline.py"
LOO_SCRIPT="${BASE_PATH}/scripts/faithfulness_exp/compute_loo_baseline.py"
LLM_SCRIPT="${BASE_PATH}/scripts/faithfulness_exp/compute_llm_baseline.py"
MAST_SCRIPT="${BASE_PATH}/scripts/faithfulness_exp/compute_mast_baseline.py"

# 结果目录
RESULTS_BASE_DIR="${BASE_PATH}/results"

# 并行执行参数（用于LOO方法）
# LOO计算是CPU密集型任务，建议设置为CPU核心数，如64核CPU可设置为64
# 0表示自动检测CPU核心数，1表示串行执行
LOO_N_THREADS=64                                # LOO计算的并行线程数

# LLM相关配置（仅用于llm和mast方法）
# 配置文件路径（如果不存在，会尝试使用环境变量OPENAI_API_KEY）
# 可以设置为具体的配置文件，如: "${BASE_PATH}/config/api_gpt-4o-mini.yaml"
# 或者设置为空字符串，使用环境变量
LLM_CONFIG_FILE="${BASE_PATH}/config/api.yaml"  # LLM配置文件路径（如果不存在，使用环境变量）
LLM_ROWS_PER_BATCH=100                          # 每批处理的行数（避免token上限）
LLM_MAX_WORKERS=10                              # 并发线程数（API调用是I/O密集型，可以设置较大值）

# 如果LLM_CONFIG_FILE不存在，可以设置环境变量（在运行脚本前设置）
# export OPENAI_API_KEY="your-api-key"
# export OPENAI_MODEL="gpt-4o"
# export OPENAI_BASE_URL="https://api.openai.com/v1"

# 日志文件
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

# 提取action table
extract_action_table() {
    local result_dir_path="$1"
    
    log "提取action table: ${result_dir_path}"
    
    # 检查是否已存在
    local action_table_file="${result_dir_path}/action_table/action_table.csv"
    if [ -f "$action_table_file" ]; then
        if [ "$FORCE_OVERWRITE" = "true" ]; then
            log "  Action table已存在，但配置为覆盖模式，将重新提取: $action_table_file"
        else
            log "  Action table已存在，跳过: $action_table_file (如需覆盖，请设置 FORCE_OVERWRITE=true)"
            return 0
        fi
    fi
    
    # 切换到项目根目录执行命令
    cd "$BASE_PATH" || {
        error "无法切换到项目根目录: $BASE_PATH"
        return 1
    }
    
    # 执行命令
    local cmd="python3 \"$EXTRACT_SCRIPT\" --result_dir \"$result_dir_path\""
    
    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        log "  ✓ Action table提取完成"
        return 0
    else
        error "  ✗ Action table提取失败"
        return 1
    fi
}

# 运行单个baseline方法
run_baseline_method() {
    local result_dir_path="$1"
    local method="$2"
    
    log "运行${method} baseline: ${result_dir_path}"
    
    # 根据方法选择脚本
    local script=""
    case "$method" in
        "random")
            script="$RANDOM_SCRIPT"
            ;;
        "loo")
            script="$LOO_SCRIPT"
            ;;
        "llm")
            script="$LLM_SCRIPT"
            ;;
        "mast")
            script="$MAST_SCRIPT"
            ;;
        *)
            error "未知的方法: $method"
            return 1
            ;;
    esac
    
    # 检查输出文件是否已存在
    local output_file="${result_dir_path}/faithfulness_exp/${method}/${method}_attribution_timeseries.csv"
    if [ -f "$output_file" ]; then
        if [ "$FORCE_OVERWRITE" = "true" ]; then
            log "  结果文件已存在，但配置为覆盖模式，将重新计算: $output_file"
        else
            log "  结果文件已存在，跳过: $output_file (如需覆盖，请设置 FORCE_OVERWRITE=true)"
            return 0
        fi
    fi
    
    # 切换到项目根目录执行命令
    cd "$BASE_PATH" || {
        error "无法切换到项目根目录: $BASE_PATH"
        return 1
    }
    
    # 构建命令
    local cmd="python3 \"$script\" --result_dir \"$result_dir_path\""
    
    # 对于loo方法，添加并行线程数参数
    if [ "$method" == "loo" ]; then
        cmd="$cmd --n-threads $LOO_N_THREADS"
    fi
    
    # 对于llm和mast方法，添加config和其他参数
    if [ "$method" == "llm" ] || [ "$method" == "mast" ]; then
        # 如果配置文件存在，使用配置文件；否则依赖环境变量
        if [ -f "$LLM_CONFIG_FILE" ]; then
            cmd="$cmd --config \"$LLM_CONFIG_FILE\""
        else
            log "  警告: LLM配置文件不存在: $LLM_CONFIG_FILE，将使用环境变量OPENAI_API_KEY"
        fi
        # 添加rows-per-batch参数
        cmd="$cmd --rows-per-batch $LLM_ROWS_PER_BATCH"
        # 添加max-workers参数（并发线程数）
        cmd="$cmd --max-workers $LLM_MAX_WORKERS"
    fi
    
    # 执行命令
    local start_time=$(date +%s)
    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "  ✓ ${method} baseline计算完成 (耗时: ${duration}秒)"
        
        # 验证输出文件是否存在
        if [ -f "$output_file" ]; then
            log "    结果文件: $output_file"
            return 0
        else
            error "    警告: 计算完成但未找到输出文件: $output_file"
            return 1
        fi
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        error "  ✗ ${method} baseline计算失败 (耗时: ${duration}秒)"
        return 1
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
    
    # 提取action table（如果需要）
    if [[ " ${BASELINE_METHODS[@]} " =~ " llm " ]] || [[ " ${BASELINE_METHODS[@]} " =~ " mast " ]]; then
        if ! extract_action_table "$result_dir_path"; then
            error "Action table提取失败，跳过该结果目录"
            return 1
        fi
    fi
    
    # 运行每个baseline方法
    local success_count=0
    local fail_count=0
    
    for method in "${BASELINE_METHODS[@]}"; do
        if run_baseline_method "$result_dir_path" "$method"; then
            success_count=$((success_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi
    done
    
    log "完成: ${model_name}/${result_dir_name} (成功: ${success_count}, 失败: ${fail_count})"
    
    if [ $fail_count -eq 0 ]; then
        return 0
    else
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
    log "批量计算Baseline方法"
    log "=========================================="
    log "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    log "配置:"
    if [ ${#MODEL_NAMES[@]} -eq 0 ]; then
        log "  模型列表: 自动识别所有模型"
    else
        log "  模型列表: ${MODEL_NAMES[*]}"
    fi
    log "  方法列表: ${BASELINE_METHODS[*]}"
    if [[ " ${BASELINE_METHODS[@]} " =~ " loo " ]]; then
        log "  LOO并行线程数: $LOO_N_THREADS"
    fi
    log "  结果目录: $RESULTS_BASE_DIR"
    log "  覆盖模式: $FORCE_OVERWRITE (true=覆盖已存在文件, false=跳过已存在文件)"
    log "  日志文件: $LOG_FILE"
    if [[ " ${BASELINE_METHODS[@]} " =~ " llm " ]] || [[ " ${BASELINE_METHODS[@]} " =~ " mast " ]]; then
    log "  LLM配置:"
    if [ -f "$LLM_CONFIG_FILE" ]; then
        log "    配置文件: $LLM_CONFIG_FILE (存在)"
    else
        log "    配置文件: $LLM_CONFIG_FILE (不存在，将使用环境变量)"
    fi
    log "    每批行数: $LLM_ROWS_PER_BATCH"
    log "    并发线程数: $LLM_MAX_WORKERS"
    if [ -z "$OPENAI_API_KEY" ] && [ ! -f "$LLM_CONFIG_FILE" ]; then
        error "  警告: 未设置OPENAI_API_KEY环境变量且配置文件不存在，LLM/MAST方法可能失败"
    fi
    fi
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
    log "批量计算完成"
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
main "$@"
