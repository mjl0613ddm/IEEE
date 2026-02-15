#!/bin/bash

###############################################################################
# 批量计算Faithfulness实验脚本
# 
# 该脚本用于批量处理指定模型的结果文件夹，计算faithfulness指标
###############################################################################

# ============================================================================
# 配置参数（可根据需要修改）
# ============================================================================

# 模型名称列表（例如: gpt-4o-mini, qwen-plus, claude-3-haiku-20240307）
# 可以指定多个模型，用空格分隔
# 如果为空，则自动识别所有模型
MODEL_NAMES=()

# 归因方法列表（可以指定多个方法，用空格分隔）
# 可选: shapley, random, llm, mast, loo
METHODS=("random" "loo" "llm" "mast" "shapley")

# 指标类型列表（可以指定多个指标类型，用空格分隔）
# 支持格式: deletion_top_n (n为任意正整数，如 deletion_top_3, deletion_top_10)
METRIC_TYPES=("deletion_top_3" "deletion_top_10")

# 并行执行参数
# 最大并发任务数（0表示不限制，1表示串行执行）
# 建议设置为CPU核心数的1-2倍，如64核CPU可设置为64-128
MAX_CONCURRENT=128

# 是否强制覆盖已存在的结果文件
# true: 覆盖已存在的文件
# false: 跳过已存在的文件（默认）
FORCE_OVERWRITE=false

# 项目根目录路径
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 脚本路径
FAITHFULNESS_SCRIPT="${BASE_PATH}/scripts/faithfulness_exp/compute_faithfulness.py"

# 结果目录
RESULTS_BASE_DIR="${BASE_PATH}/results"

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
    
    if [ ! -d "$RESULTS_BASE_DIR" ]; then
        error "结果目录不存在: $RESULTS_BASE_DIR"
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

# 处理单个子文件夹的单个方法的单个指标类型（内部函数，用于并行执行）
_process_faithfulness_method_metric_internal() {
    local model_name="$1"
    local subfolder="$2"
    local method="$3"
    local metric_type="$4"
    local task_id="${model_name}/${subfolder}/${method}/${metric_type}"
    local result_dir_path="${RESULTS_BASE_DIR}/${model_name}/${subfolder}"
    
    # 使用临时日志文件避免输出混乱
    local temp_log="${LOG_DIR}/temp_${TIMESTAMP}_${task_id//\//_}.log"
    
    {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始处理: ${task_id}"
        
        # 检查子文件夹是否存在
        if [ ! -d "$result_dir_path" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: 子文件夹不存在: $result_dir_path" >&2
            exit 1
        fi
        
        # 检查方法所需的数据文件是否存在
        case "$method" in
            "shapley")
                if [ ! -d "${result_dir_path}/shapley" ]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Shapley目录不存在: ${result_dir_path}/shapley" >&2
                    exit 1
                fi
                ;;
            "random"|"loo"|"llm"|"mast")
                if [ ! -d "${result_dir_path}/faithfulness_exp/${method}" ]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: ${method}方法的数据目录不存在: ${result_dir_path}/faithfulness_exp/${method}" >&2
                    exit 1
                fi
                ;;
            *)
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: 未知的方法: $method" >&2
                exit 1
                ;;
        esac
        
        # 检查结果文件是否已存在
        local result_file="${result_dir_path}/faithfulness_exp/faithfulness_results_${method}_${metric_type}.json"
        if [ -f "$result_file" ]; then
            if [ "$FORCE_OVERWRITE" = "true" ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 结果文件已存在，但配置为覆盖模式，将重新计算: $result_file"
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 结果文件已存在，跳过: $result_file"
                exit 0
            fi
        fi
        
        # 构建命令
        local cmd="python3 \"$FAITHFULNESS_SCRIPT\""
        cmd="${cmd} --result_dir \"$result_dir_path\""
        cmd="${cmd} --method \"$method\""
        cmd="${cmd} --metric_type \"$metric_type\""
        
        # 切换到项目根目录执行命令
        cd "$BASE_PATH" || {
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: 无法切换到项目根目录: $BASE_PATH" >&2
            exit 1
        }
        
        # 执行命令
        local start_time=$(date +%s)
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 执行命令: $cmd"
        
        if eval "$cmd" >&2 2>&1; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ ${task_id} 完成 (耗时: ${duration}秒)"
            exit 0
        else
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ ${task_id} 失败 (耗时: ${duration}秒)" >&2
            exit 1
        fi
    } > "$temp_log" 2>&1
    
    local exit_code=$?
    
    # 将临时日志追加到主日志文件
    {
        echo "--- Task: ${task_id} ---"
        cat "$temp_log"
        echo "--- End Task: ${task_id} ---"
    } >> "$LOG_FILE"
    
    # 删除临时日志文件
    rm -f "$temp_log"
    
    return $exit_code
}

# 处理单个子文件夹的单个方法的单个指标类型（兼容原有接口）
process_faithfulness_method_metric() {
    _process_faithfulness_method_metric_internal "$@"
}

# 处理单个子文件夹的单个方法（所有指标类型）
process_faithfulness_method() {
    local model_name="$1"
    local subfolder="$2"
    local method="$3"
    local result_dir_path="${RESULTS_BASE_DIR}/${model_name}/${subfolder}"
    
    log "  - 处理 ${method} 方法..."
    
    # 检查子文件夹是否存在
    if [ ! -d "$result_dir_path" ]; then
        error "    子文件夹不存在: $result_dir_path"
        return 1
    fi
    
    # 检查方法所需的数据文件是否存在（只检查一次）
    case "$method" in
        "shapley")
            if [ ! -d "${result_dir_path}/shapley" ]; then
                error "    Shapley目录不存在: ${result_dir_path}/shapley，跳过该方法"
                return 1
            fi
            ;;
        "random"|"loo"|"llm"|"mast")
            if [ ! -d "${result_dir_path}/faithfulness_exp/${method}" ]; then
                error "    ${method}方法的数据目录不存在: ${result_dir_path}/faithfulness_exp/${method}，跳过该方法（需要先运行compute_*_baseline.py）"
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
    local result_dir_path="${RESULTS_BASE_DIR}/${model_name}/${subfolder}"
    
    log "=========================================="
    log "处理: ${model_name}/${subfolder}"
    log "=========================================="
    
    # 检查子文件夹是否存在
    if [ ! -d "$result_dir_path" ]; then
        error "子文件夹不存在: $result_dir_path"
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

# 并行执行任务列表
parallel_execute_tasks() {
    local task_list=("$@")
    local total_tasks=${#task_list[@]}
    local max_concurrent=$MAX_CONCURRENT
    
    if [ $total_tasks -eq 0 ]; then
        return 0
    fi
    
    log "开始并行执行 ${total_tasks} 个任务 (最大并发数: ${max_concurrent})"
    
    local success_count=0
    local fail_count=0
    local completed_count=0
    local task_index=0
    
    # 如果MAX_CONCURRENT为0或1，串行执行
    if [ "$max_concurrent" -le 1 ]; then
        log "使用串行执行模式"
        for task in "${task_list[@]}"; do
            IFS='|' read -r model_name subfolder method metric_type <<< "$task"
            if _process_faithfulness_method_metric_internal "$model_name" "$subfolder" "$method" "$metric_type"; then
                success_count=$((success_count + 1))
            else
                fail_count=$((fail_count + 1))
            fi
            completed_count=$((completed_count + 1))
            if [ $((completed_count % 10)) -eq 0 ] || [ $completed_count -eq $total_tasks ]; then
                log "进度: ${completed_count}/${total_tasks} (成功: ${success_count}, 失败: ${fail_count})"
            fi
        done
    else
        # 并行执行模式
        local pids=()
        
        while [ $task_index -lt $total_tasks ] || [ ${#pids[@]} -gt 0 ]; do
            # 启动新任务直到达到最大并发数
            while [ ${#pids[@]} -lt $max_concurrent ] && [ $task_index -lt $total_tasks ]; do
                task="${task_list[$task_index]}"
                IFS='|' read -r model_name subfolder method metric_type <<< "$task"
                
                # 后台执行任务
                _process_faithfulness_method_metric_internal "$model_name" "$subfolder" "$method" "$metric_type" &
                local pid=$!
                
                pids+=($pid)
                task_index=$((task_index + 1))
            done
            
            # 等待任意一个任务完成（使用循环检查进程状态）
            if [ ${#pids[@]} -gt 0 ]; then
                local new_pids=()
                for pid in "${pids[@]}"; do
                    if kill -0 "$pid" 2>/dev/null; then
                        # 进程还在运行
                        new_pids+=($pid)
                    else
                        # 进程已结束，等待获取退出码
                        wait "$pid"
                        local task_exit_code=$?
                        if [ $task_exit_code -eq 0 ]; then
                            success_count=$((success_count + 1))
                        else
                            fail_count=$((fail_count + 1))
                        fi
                        completed_count=$((completed_count + 1))
                        
                        if [ $((completed_count % 10)) -eq 0 ] || [ $completed_count -eq $total_tasks ]; then
                            log "进度: ${completed_count}/${total_tasks} (成功: ${success_count}, 失败: ${fail_count}, 运行中: ${#new_pids[@]})"
                        fi
                    fi
                done
                pids=("${new_pids[@]}")
            fi
            
            # 短暂休眠避免CPU占用过高
            if [ ${#pids[@]} -gt 0 ]; then
                sleep 0.1
            fi
        done
    fi
    
    log "并行执行完成: 总共 ${total_tasks} 个任务 (成功: ${success_count}, 失败: ${fail_count})"
    
    if [ $fail_count -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

# 处理单个模型的所有子文件夹
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
    
    # 收集所有任务
    local task_list=()
    for subfolder in "${subfolders[@]}"; do
        local result_dir_path="${RESULTS_BASE_DIR}/${model_name}/${subfolder}"
        
        # 检查子文件夹是否存在
        if [ ! -d "$result_dir_path" ]; then
            continue
        fi
        
        for method in "${METHODS[@]}"; do
            # 检查方法所需的数据文件是否存在
            case "$method" in
                "shapley")
                    if [ ! -d "${result_dir_path}/shapley" ]; then
                        continue
                    fi
                    ;;
                "random"|"loo"|"llm"|"mast")
                    if [ ! -d "${result_dir_path}/faithfulness_exp/${method}" ]; then
                        continue
                    fi
                    ;;
                *)
                    continue
                    ;;
            esac
            
            for metric_type in "${METRIC_TYPES[@]}"; do
                # 检查结果文件是否已存在
                local result_file="${result_dir_path}/faithfulness_exp/faithfulness_results_${method}_${metric_type}.json"
                if [ -f "$result_file" ] && [ "$FORCE_OVERWRITE" != "true" ]; then
                    continue
                fi
                
                # 添加到任务列表（使用|作为分隔符）
                task_list+=("${model_name}|${subfolder}|${method}|${metric_type}")
            done
        done
    done
    
    if [ ${#task_list[@]} -eq 0 ]; then
        log "  没有需要处理的任务（所有任务都已完成或缺少数据）"
        return 0
    fi
    
    log "  收集到 ${#task_list[@]} 个任务"
    
    # 并行执行所有任务
    if parallel_execute_tasks "${task_list[@]}"; then
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
    log "批量计算Faithfulness实验"
    log "=========================================="
    log "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 如果未指定模型，自动发现
    local model_list=()
    if [ ${#MODEL_NAMES[@]} -eq 0 ]; then
        log "未指定模型列表，自动发现所有模型目录..."
        for item in "$RESULTS_BASE_DIR"/*; do
            if [ -d "$item" ]; then
                local model_name=$(basename "$item")
                # 跳过汇总目录
                if [ "$model_name" != "risk_feature" ] && [ "$model_name" != "faithfulness_exp" ]; then
                    model_list+=("$model_name")
                fi
            fi
        done
        if [ ${#model_list[@]} -eq 0 ]; then
            error "未找到任何模型目录"
            exit 1
        fi
        log "发现 ${#model_list[@]} 个模型目录: ${model_list[*]}"
        log ""
    else
        # 使用指定的模型列表
        model_list=("${MODEL_NAMES[@]}")
    fi
    
    log "配置:"
    log "  模型列表: ${model_list[*]}"
    log "  方法列表: ${METHODS[*]}"
    log "  指标类型列表: ${METRIC_TYPES[*]}"
    log "  最大并发数: $MAX_CONCURRENT"
    log "  覆盖模式: $FORCE_OVERWRITE (true=覆盖已存在文件, false=跳过已存在文件)"
    log "  结果目录: $RESULTS_BASE_DIR"
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
    for model_name in "${model_list[@]}"; do
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
