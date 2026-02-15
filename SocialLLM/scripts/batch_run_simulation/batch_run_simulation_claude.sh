#!/bin/bash

###############################################################################
# 批量运行SocialLLM模拟脚本
# 
# 该脚本用于批量使用不同模型运行SocialLLM模拟
# 每个模型使用不同seed运行15次（seed 42-56）
###############################################################################

# ============================================================================
# 配置参数（可根据需要修改）
# ============================================================================

# 运行名称列表（用于结果子目录命名，共用一个 API 配置 config/api.yaml）
MODEL_NAMES=("run1")

# 固定参数
NUM_AGENTS=20
NUM_STEPS=30
SEED_START=57
SEED_COUNT=15  # 运行15次

# 并发数（API调用是I/O密集型，与CPU核心数无关）
MAX_CONCURRENT=30  # 默认并发数，可根据API服务器容量调整

# 项目根目录路径
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 主配置文件路径
MAIN_CONFIG="${BASE_PATH}/config/config.yaml"

# 主程序脚本
MAIN_SCRIPT="${BASE_PATH}/main.py"

# 结果目录（基础路径）
RESULTS_BASE_DIR="${BASE_PATH}/results"

# 日志文件
LOG_DIR="${BASE_PATH}/scripts/batch_run_simulation/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/batch_run_simulation_${TIMESTAMP}.log"

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
    
    if [ ! -f "$MAIN_SCRIPT" ]; then
        error "主程序脚本不存在: $MAIN_SCRIPT"
        missing_files=$((missing_files + 1))
    fi
    
    if [ ! -f "$MAIN_CONFIG" ]; then
        error "主配置文件不存在: $MAIN_CONFIG"
        missing_files=$((missing_files + 1))
    fi
    
    if [ $missing_files -gt 0 ]; then
        error "缺少必要文件，退出"
        return 1
    fi
    
    return 0
}

# 创建临时配置文件（修改llm.config_path和simulation.seed）
create_temp_config() {
    local api_config_file="$1"
    local seed="$2"
    local temp_config=$(mktemp)
    
    # 将API配置文件路径转换为相对于BASE_PATH的路径
    local api_config_rel_path="${api_config_file#$BASE_PATH/}"
    
    # 使用python或sed修改配置文件
    # 这里使用python来修改yaml文件（更安全）
    python3 << EOF
import yaml
import sys
import os

# 读取原始配置
with open('$MAIN_CONFIG', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 修改配置
config['simulation']['num_agents'] = $NUM_AGENTS
config['simulation']['num_steps'] = $NUM_STEPS
config['simulation']['seed'] = $seed
config['llm']['config_path'] = '$api_config_rel_path'

# 写入临时文件
with open('$temp_config', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
EOF
    
    if [ $? -ne 0 ]; then
        error "创建临时配置文件失败"
        rm -f "$temp_config"
        return 1
    fi
    
    echo "$temp_config"
    return 0
}

# 运行单个模拟任务
run_simulation() {
    local model_name="$1"
    local seed="$2"
    local output_dir="${RESULTS_BASE_DIR}/${model_name}/${model_name}_${seed}"
    local api_config_file="${BASE_PATH}/config/api.yaml"
    local results_file="${output_dir}/results.json"
    
    # 检查结果是否已存在
    if [ -f "$results_file" ]; then
        log "    跳过: ${model_name} (seed=${seed}) - 结果已存在"
        return 0
    fi
    
    log "    开始运行: ${model_name} (seed=${seed})"
    
    # 检查API配置文件是否存在
    if [ ! -f "$api_config_file" ]; then
        error "    API配置文件不存在: $api_config_file（请从 config/api_example.yaml 复制为 config/api.yaml 并填写 key），跳过"
        return 1
    fi
    
    # 创建临时配置文件
    local temp_config=$(create_temp_config "$api_config_file" "$seed")
    if [ -z "$temp_config" ]; then
        error "    创建临时配置文件失败，跳过"
        return 1
    fi
    
    # 切换到项目根目录执行命令
    cd "$BASE_PATH" || {
        error "    无法切换到项目根目录: $BASE_PATH"
        rm -f "$temp_config"
        return 1
    }
    
    # 执行命令
    local start_time=$(date +%s)
    local cmd="python3 \"$MAIN_SCRIPT\" --config \"$temp_config\" --output \"$output_dir\" --mode simulate"
    
    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "    ✓ ${model_name} (seed=${seed}) 完成 (耗时: ${duration}秒)"
        rm -f "$temp_config"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        error "    ✗ ${model_name} (seed=${seed}) 失败 (耗时: ${duration}秒)"
        rm -f "$temp_config"
        return 1
    fi
}

# 处理单个模型的所有seed
process_model() {
    local model_name="$1"
    local api_config_file="${BASE_PATH}/config/api.yaml"
    
    log "=========================================="
    log "处理模型: ${model_name}"
    log "=========================================="
    
    # 检查API配置文件是否存在
    if [ ! -f "$api_config_file" ]; then
        error "API配置文件不存在: $api_config_file（请从 config/api_example.yaml 复制为 config/api.yaml），跳过"
        return 1
    fi
    
    local success_count=0
    local fail_count=0
    local skip_count=0
    local all_seeds=()
    
    # 生成所有seed列表
    for i in $(seq 0 $((SEED_COUNT - 1))); do
        all_seeds+=($((SEED_START + i)))
    done
    
    # 分批处理：每次启动MAX_CONCURRENT个任务，等待完成后再启动下一批
    local total_seeds=${#all_seeds[@]}
    local current_index=0
    
    while [ $current_index -lt $total_seeds ]; do
        local batch_pids=()
        local batch_seeds=()
        local batch_count=0
        
        # 启动一批任务
        while [ $batch_count -lt $MAX_CONCURRENT ] && [ $current_index -lt $total_seeds ]; do
            local seed=${all_seeds[$current_index]}
            local output_dir="${RESULTS_BASE_DIR}/${model_name}/${model_name}_${seed}"
            local results_file="${output_dir}/results.json"
            
            # 检查是否已存在结果
            if [ -f "$results_file" ]; then
                log "  跳过任务: ${model_name} (seed=${seed}) - 结果已存在"
                success_count=$((success_count + 1))
                skip_count=$((skip_count + 1))
                current_index=$((current_index + 1))
                continue
            fi
            
            log "  启动任务: ${model_name} (seed=${seed}), 批次进度: $((current_index + 1))/${total_seeds}"
            
            # 后台运行任务
            run_simulation "$model_name" "$seed" &
            local pid=$!
            batch_pids+=($pid)
            batch_seeds+=($seed)
            batch_count=$((batch_count + 1))
            current_index=$((current_index + 1))
        done
        
        # 等待这一批任务完成
        for j in "${!batch_pids[@]}"; do
            local pid=${batch_pids[$j]}
            local seed=${batch_seeds[$j]}
            wait "$pid" 2>/dev/null
            local exit_code=$?
            if [ $exit_code -eq 0 ]; then
                success_count=$((success_count + 1))
            else
                fail_count=$((fail_count + 1))
            fi
        done
    done
    
    if [ $skip_count -gt 0 ]; then
        log "完成: ${model_name} (成功: ${success_count}, 失败: ${fail_count}, 跳过: ${skip_count})"
    else
        log "完成: ${model_name} (成功: ${success_count}, 失败: ${fail_count})"
    fi
    
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
    log "批量运行SocialLLM模拟"
    log "=========================================="
    log "开始时间: $(date '+%Y-%m-%d %H:%M%S')"
    log "配置:"
    log "  模型列表: ${MODEL_NAMES[*]}"
    log "  固定参数: ${NUM_AGENTS} agents, ${NUM_STEPS} steps"
    log "  Seed范围: ${SEED_START} 到 $((SEED_START + SEED_COUNT - 1)) (共${SEED_COUNT}次)"
    log "  并发数: ${MAX_CONCURRENT}"
    log "  结果目录: $RESULTS_BASE_DIR"
    log "  日志文件: $LOG_FILE"
    log ""
    
    # 检查依赖
    if ! check_dependencies; then
        error "依赖检查失败，退出"
        exit 1
    fi
    
    # 创建结果基础目录
    mkdir -p "$RESULTS_BASE_DIR"
    
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
