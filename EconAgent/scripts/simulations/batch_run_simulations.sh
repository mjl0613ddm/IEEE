#!/bin/bash
# 批量运行不同模型和种子的模拟，并生成图片
# 使用方法: 
#   cd EconAgent/scripts
#   ./simulations/batch_run_simulations.sh
# 或者从EconAgent根目录运行:
#   ./scripts/simulations/batch_run_simulations.sh

# 注意：不使用 set -e，因为我们需要在单个运行失败时继续执行其他运行

# 设置工作目录
# 脚本在 scripts/simulations/ 目录下，需要切换到项目根目录（EconAgent）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# 配置（相对于项目根目录）
CONFIG_FILE="config.yaml"
SIMULATE_SCRIPT="simulate.py"
PLOT_SCRIPT="scripts/plot/plot_world_metrics.py"
DATA_ROOT="./datas"

# 模型配置（模型名称 -> config.yaml中的模型标识符）
declare -A MODEL_MAP=(
    # ["gpt"]="gpt-4o-mini"
    # ["llama"]="llama-3.1-8b-instruct"
    # ["claude"]="claude-3-haiku-20240307"
    # ["ds"]="deepseek-v3.2"
    ["qwen"]="qwen-plus"
)

# 种子列表（跳过42，因为已有结果）
SEEDS=(57 58 59 60 61 62)

# 模拟参数
NUM_AGENTS=10
EPISODE_LENGTH=50

# 并行任务数（可通过环境变量 MAX_PARALLEL_JOBS 覆盖，默认2）
# 注意：每个模拟任务内部还会使用多进程（默认15个），所以总并发数 = MAX_PARALLEL_JOBS × 15
# 例如：MAX_PARALLEL_JOBS=2 时，总共有约 2×15=30 个进程
# 根据系统内存和CPU核心数调整：
#   - 内存 < 8GB: 建议 1-2
#   - 内存 8-16GB: 建议 2-4
#   - 内存 > 16GB: 建议 4-8
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-3}

# 备份原始配置文件
CONFIG_BACKUP="${CONFIG_FILE}.backup"
if [ ! -f "$CONFIG_BACKUP" ]; then
    cp "$CONFIG_FILE" "$CONFIG_BACKUP"
    echo "✅ 已备份配置文件: $CONFIG_BACKUP"
fi

# 函数：修改配置文件（使用临时配置文件避免并行冲突）
update_config() {
    local model_name=$1
    local run_name=$2
    local seed=$3
    local num_agents=$4
    local episode_length=$5
    local temp_config_file=$6  # 临时配置文件路径
    
    # 使用Python脚本来修改YAML文件（更安全）
    python3 << EOF
import yaml
import sys
import os
import shutil

config_file = "$CONFIG_FILE"
temp_config_file = "$temp_config_file"
model_name = "$model_name"
run_name = "$run_name"
seed = int("$seed")
num_agents = int("$num_agents")
episode_length = int("$episode_length")

# 从备份文件读取原始配置（如果临时文件不存在）
if not os.path.exists(temp_config_file):
    # 优先使用备份文件，如果不存在则使用原始配置文件
    if os.path.exists("${CONFIG_BACKUP}"):
        source_file = "${CONFIG_BACKUP}"
    elif os.path.exists(config_file):
        source_file = config_file
    else:
        print(f"错误: 找不到配置文件: {config_file} 或备份文件: ${CONFIG_BACKUP}", file=sys.stderr)
        sys.exit(1)
    shutil.copy(source_file, temp_config_file)

# 读取配置文件
with open(temp_config_file, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 更新配置
config['llm']['model'] = model_name
config['simulation']['run_name'] = run_name
config['simulation']['seed'] = seed
config['simulation']['num_agents'] = num_agents
config['simulation']['episode_length'] = episode_length

# 写回临时文件
with open(temp_config_file, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

print(f"✅ 已更新配置: model={model_name}, run_name={run_name}, seed={seed}, agents={num_agents}, length={episode_length}")
EOF
}

# 函数：恢复配置文件
restore_config() {
    local quiet=${1:-0}  # 默认为0（不安静模式）
    if [ -f "$CONFIG_BACKUP" ]; then
        cp "$CONFIG_BACKUP" "$CONFIG_FILE"
        if [ "$quiet" -eq 0 ]; then
            echo "✅ 已恢复配置文件"
        fi
    fi
}

# 函数：运行模拟
run_simulation() {
    local model_short=$1
    local seed=$2
    local model_family="$model_short"  # 模型家族名称（model_short就是家族名称）
    local run_name_simple="${model_short}_${seed}"  # 简单的run_name用于日志（使用下划线）
    local run_name="${model_family}/${run_name_simple}"  # 包含路径的run_name用于配置
    local success=0
    
    # 创建临时配置文件，避免并行任务冲突
    local temp_config_file="${CONFIG_FILE}.tmp.${run_name_simple}.$$"
    
    echo ""
    echo "=========================================="
    echo "🚀 开始运行: $run_name_simple"
    echo "=========================================="
    
    # 检查是否已经存在结果（使用新的文件夹结构）
    local data_folder="${DATA_ROOT}/${run_name}"
    local csv_file="${data_folder}/metrics_csv/world_metrics.csv"
    local plot_dir="${data_folder}/plot"
    
    if [ -d "$data_folder" ] && [ -f "$csv_file" ]; then
        echo "⏭️  跳过: $run_name_simple 的结果已存在"
        
        # 检查图片是否存在，如果不存在则生成
        local price_plot="${plot_dir}/price_inflation_rate.png"
        local risk_plot="${plot_dir}/risk_indicator_naive.png"
        
        if [ ! -f "$price_plot" ] || [ ! -f "$risk_plot" ]; then
            echo "📈 生成缺失的图片..."
            if [ -f "$csv_file" ]; then
                # 生成 price_inflation_rate 图片
                if [ ! -f "$price_plot" ]; then
                    python3 "$PLOT_SCRIPT" --csv-file "$csv_file" \
                        price_inflation_rate 2>/dev/null || echo "⚠️  生成 price_inflation_rate 图片失败"
                fi
                
                # 生成 risk_indicator_naive 图片
                if [ ! -f "$risk_plot" ]; then
                    python3 "$PLOT_SCRIPT" --csv-file "$csv_file" \
                        risk_indicator_naive 2>/dev/null || echo "⚠️  生成 risk_indicator_naive 图片失败"
                fi
                
                echo "✅ 图片生成完成: $run_name_simple"
            fi
        else
            echo "✅ 图片已存在，无需重新生成"
        fi
        
        echo "✅ 跳过: $run_name_simple (结果已存在)"
        echo ""
        return 0
    fi
    
    # 更新配置（使用临时配置文件）
    local model_full="${MODEL_MAP[$model_short]}"
    update_config "$model_full" "$run_name" "$seed" "$NUM_AGENTS" "$EPISODE_LENGTH" "$temp_config_file"
    
    # 将临时配置文件复制到config.yaml（使用文件锁确保原子性）
    # 使用flock确保同一时间只有一个任务在更新config.yaml
    (
        flock -w 30 200 || {
            echo "⚠️  警告: 无法获取配置文件锁，等待中..."
            flock 200
        }
        cp "$temp_config_file" "$CONFIG_FILE"
    ) 200>"${CONFIG_FILE}.lock"
    
    # 运行模拟（通过环境变量传递配置文件路径，避免冲突）
    echo "📊 运行模拟..."
    # 使用环境变量传递配置文件路径，让每个任务使用自己的配置
    ECONAGENT_CONFIG_FILE="$temp_config_file" python3 "$SIMULATE_SCRIPT" --seed "$seed" --policy_model gpt --num_agents "$NUM_AGENTS" --episode_length "$EPISODE_LENGTH"
    local sim_exit_code=$?
    
    if [ $sim_exit_code -eq 0 ]; then
        echo "✅ 模拟完成: $run_name_simple"
        success=1
    else
        echo "❌ 模拟失败: $run_name_simple"
        success=0
    fi
    
    # 生成图片（即使模拟失败也尝试生成，以防部分数据已生成）
    if [ $success -eq 1 ]; then
        echo "📈 生成图片..."
        # 使用新的文件夹结构
        local data_folder="${DATA_ROOT}/${run_name}"
        
        if [ -d "$data_folder" ]; then
            # 检查CSV文件是否存在
            local csv_file="${data_folder}/metrics_csv/world_metrics.csv"
            if [ -f "$csv_file" ]; then
                # 生成 price_inflation_rate 图片
                python3 "$PLOT_SCRIPT" --csv-file "$csv_file" \
                    price_inflation_rate 2>/dev/null || echo "⚠️  生成 price_inflation_rate 图片失败"
                
                # 生成 risk_indicator_naive 图片
                python3 "$PLOT_SCRIPT" --csv-file "$csv_file" \
                    risk_indicator_naive 2>/dev/null || echo "⚠️  生成 risk_indicator_naive 图片失败"
                
                echo "✅ 图片生成完成: $run_name_simple"
            else
                echo "⚠️  警告: CSV文件不存在: $csv_file"
            fi
        else
            echo "⚠️  警告: 数据目录不存在: $data_folder"
        fi
    fi
    
    # 清理临时配置文件
    if [ -f "$temp_config_file" ]; then
        rm -f "$temp_config_file"
    fi
    
    # 恢复配置文件（使用文件锁确保安全）
    (
        flock -w 30 200 || flock 200
        restore_config 1  # 安静模式
    ) 200>"${CONFIG_FILE}.lock"
    
    if [ $success -eq 1 ]; then
        echo "✅ 完成: $run_name_simple"
    else
        echo "❌ 失败: $run_name_simple"
    fi
    echo ""
    
    return $success
}

# 主循环
echo "=========================================="
echo "批量运行模拟脚本（并行模式）"
echo "=========================================="
echo "模型: ${!MODEL_MAP[@]}"
echo "种子: ${SEEDS[@]}"
echo "Agent数量: $NUM_AGENTS"
echo "模拟长度: $EPISODE_LENGTH 个月"
echo "最大并行任务数: $MAX_PARALLEL_JOBS"
echo ""

# 捕获中断信号，确保恢复配置
trap restore_config EXIT INT TERM

# 准备所有任务
declare -a task_queue=()
for model_short in "${!MODEL_MAP[@]}"; do
    for seed in "${SEEDS[@]}"; do
        task_queue+=("${model_short}:${seed}")
    done
done

total_tasks=${#task_queue[@]}
echo "总任务数: $total_tasks"
echo ""

# 并行执行任务
failed_runs=()
completed_tasks=0
declare -A running_jobs=()  # job_id -> "model:seed"

# 函数：启动一个任务
start_task() {
    local task=$1
    local model_short="${task%%:*}"
    local seed="${task##*:}"
    
    # 在后台运行任务
    (
        run_simulation "$model_short" "$seed"
    ) &
    
    local job_pid=$!
    running_jobs[$job_pid]="$task"
    echo "🚀 启动任务: ${model_short}-${seed} (PID: $job_pid)"
}

# 函数：等待任务完成并处理结果
wait_for_job() {
    local job_pid=$1
    local task="${running_jobs[$job_pid]}"
    unset running_jobs[$job_pid]
    
    wait $job_pid
    local exit_code=$?
    
    completed_tasks=$((completed_tasks + 1))
    local model_short="${task%%:*}"
    local seed="${task##*:}"
    local run_name_simple="${model_short}_${seed}"
    
    if [ $exit_code -ne 0 ]; then
        failed_runs+=("$run_name_simple")
        echo "❌ 任务失败: $run_name_simple (进度: $completed_tasks/$total_tasks)"
    else
        echo "✅ 任务完成: $run_name_simple (进度: $completed_tasks/$total_tasks)"
    fi
}

# 主执行循环
task_index=0
while [ $task_index -lt $total_tasks ] || [ ${#running_jobs[@]} -gt 0 ]; do
    # 启动新任务（如果还有待处理的任务且未达到最大并行数）
    while [ $task_index -lt $total_tasks ] && [ ${#running_jobs[@]} -lt $MAX_PARALLEL_JOBS ]; do
        start_task "${task_queue[$task_index]}"
        task_index=$((task_index + 1))
    done
    
    # 等待任意一个任务完成
    if [ ${#running_jobs[@]} -gt 0 ]; then
        # 轮询检查任务状态（兼容性更好）
        sleep 2  # 等待一段时间，避免过于频繁的检查
        found_completed=false
        for pid in "${!running_jobs[@]}"; do
            if ! kill -0 $pid 2>/dev/null; then
                # 进程已结束
                wait_for_job $pid
                found_completed=true
                break
            fi
        done
        # 如果没有找到完成的任务，继续等待
        if [ "$found_completed" = false ]; then
            sleep 1
        fi
    fi
done

# 等待所有剩余任务完成
for pid in "${!running_jobs[@]}"; do
    wait $pid 2>/dev/null
    wait_for_job $pid
done

# 最终恢复配置文件（确保恢复）
restore_config

# 报告结果
echo ""
echo "=========================================="
if [ ${#failed_runs[@]} -eq 0 ]; then
    echo "✅ 所有运行都成功完成！"
else
    echo "⚠️  以下运行失败 (${#failed_runs[@]}/${total_tasks}):"
    for run in "${failed_runs[@]}"; do
        echo "   - $run"
    done
fi

echo "=========================================="
echo "🎉 所有任务完成！"
echo "=========================================="

