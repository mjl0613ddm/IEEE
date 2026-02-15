#!/bin/bash

length=20
model='gpt-4o-mini'
activate_prob="0.8"
log_dir="logs_${length}_nips_${model}_${activate_prob}"

# 新的文件结构：所有输出在 results/ 目录下
results_dir="results"
full_log_dir="${results_dir}/${log_dir}"

# 创建 results 目录（如果不存在）
mkdir -p "$results_dir"

# 注意：不再删除整个目录，因为 simulation.py 会自动创建
# 如果需要清理，可以手动删除：rm -rf "$full_log_dir"

# 检查并自动设置数据库文件（现在在 results/ 下）
user_db_path="${full_log_dir}/user_${length}.db"
forum_db_path="${full_log_dir}/forum_${length}.db"

echo "=== 自动数据库设置 ==="

# 确保输出目录存在（在复制数据库之前）
mkdir -p "$full_log_dir"

# 检查用户数据库
if [[ ! -f "$user_db_path" || $(stat -f%z "$user_db_path" 2>/dev/null || stat -c%s "$user_db_path" 2>/dev/null) -eq 0 ]]; then
    echo "用户数据库不存在或为空，从模板复制..."
    if [[ -f "data/sys_${length}.db" ]]; then
        cp "data/sys_${length}.db" "$user_db_path"
        echo "✅ 复制用户数据库: sys_${length}.db → user_${length}.db"
    else
        echo "❌ 找不到模板数据库: data/sys_${length}.db"
        exit 1
    fi
else
    echo "✅ 用户数据库已存在: $user_db_path"
fi

# 论坛数据库初始化交由 simulation.py 在非 checkpoint 模式下处理

echo "===================="

timestamp=$(date +%Y%m%d_%H%M%S)
log_file="${full_log_dir}/simulation_${timestamp}.log"

# Run simulation (同时输出到终端和日志)
echo "Simulation starting..."
echo "Output directory: ${full_log_dir}"
echo "Log file: ${log_file}"
echo "===================="

# 注意：simulation.py 会自动将 log_dir 改为 results/{log_dir}
# 所以这里传入原始的 log_dir 名称即可
python simulation.py \
    --log_dir $log_dir \
    --user_db "${full_log_dir}/user_${length}.db" \
    --forum_db "${full_log_dir}/forum_${length}.db" \
    --user_graph_save_name "user_graph_${log_dir}" \
    --start_date "2023-06-15" \
    --end_date "2023-8-15" \
    --max_workers 250 \
    --config_path './config/api.yaml' \
    --activate_prob ${activate_prob} \
    --belief_init_path "util/belief/belief_${length}.csv" \
    2>&1 | tee "${log_file}"

echo "Simulation completed!"