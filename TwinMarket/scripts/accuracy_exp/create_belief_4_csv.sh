#!/bin/bash

# 从 belief_1000.csv 中提取4个用户创建 belief_4.csv
# 需要先运行 create_sys_4_db.sh 来获取选择的用户ID

SOURCE_CSV="util/belief/belief_1000.csv"
TARGET_CSV="util/belief/belief_4.csv"
SOURCE_DB="data/sys_4.db"

echo "=== 从 ${SOURCE_CSV} 创建 ${TARGET_CSV} ==="

# 检查源文件是否存在
if [ ! -f "$SOURCE_CSV" ]; then
    echo "❌ 源文件不存在: $SOURCE_CSV"
    echo "   尝试使用其他源文件..."
    
    # 尝试其他可能的源文件
    if [ -f "util/belief/belief_1000_0129.csv" ]; then
        SOURCE_CSV="util/belief/belief_1000_0129.csv"
        echo "   使用: $SOURCE_CSV"
    elif [ -f "util/belief/belief_100.csv" ]; then
        SOURCE_CSV="util/belief/belief_100.csv"
        echo "   使用: $SOURCE_CSV"
    else
        echo "❌ 找不到任何可用的belief源文件"
        exit 1
    fi
fi

# 检查数据库是否存在（用于获取用户ID列表）
if [ ! -f "$SOURCE_DB" ]; then
    echo "⚠️  警告: 数据库 $SOURCE_DB 不存在"
    echo "   请先运行: scripts/accuracy_exp/create_sys_4_db.sh"
    exit 1
fi

# 使用 Python 来确保准确匹配
echo "📋 从数据库获取用户ID列表并匹配..."
python3 << PYTHON_SCRIPT
import csv
import sqlite3
import sys

source_csv = "${SOURCE_CSV}"
target_csv = "${TARGET_CSV}"
source_db = "${SOURCE_DB}"

# 获取数据库中的用户ID
conn = sqlite3.connect(source_db)
cur = conn.cursor()
cur.execute('SELECT user_id FROM Profiles ORDER BY user_id')
db_users = [str(row[0]) for row in cur.fetchall()]
conn.close()

print(f"数据库中的用户ID: {db_users}")

# 读取源CSV并创建字典
belief_dict = {}
with open(source_csv, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    
    for row in reader:
        if len(row) >= 1:
            user_id = row[0]
            belief_dict[user_id] = row

print(f"源文件中找到 {len(belief_dict)} 个用户的belief数据")

# 创建 belief_4.csv
with open(target_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)  # header
    
    found_count = 0
    for user_id in db_users:
        if user_id in belief_dict:
            writer.writerow(belief_dict[user_id])
            found_count += 1
        else:
            print(f'⚠️  警告: 用户 {user_id} 在 belief源文件中未找到', file=sys.stderr)

print(f'✅ 匹配完成: {found_count}/{len(db_users)} 个用户')
PYTHON_SCRIPT

# 验证结果
if [ -f "$TARGET_CSV" ]; then
    # 使用Python正确统计CSV行数（考虑belief字段中的换行符）
    USER_COUNT=$(python3 << PYTHON_SCRIPT 2>/dev/null
import csv
import sys

target_csv = "${TARGET_CSV}"
try:
    with open(target_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过header
        count = sum(1 for row in reader)
    print(count, end='')
except Exception as e:
    print("0", end='')
    sys.exit(1)
PYTHON_SCRIPT
)
    
    # 如果USER_COUNT为空或非数字，设置为0
    if [ -z "$USER_COUNT" ] || ! [[ "$USER_COUNT" =~ ^[0-9]+$ ]]; then
        USER_COUNT=0
    fi
    
    echo ""
    echo "✅ 创建完成！"
    echo "   - 目标文件: $TARGET_CSV"
    echo "   - 用户数量: $USER_COUNT / 4"
    
    if [ "$USER_COUNT" -eq 4 ]; then
        echo "✅ 验证通过！"
    else
        echo "⚠️  警告: 用户数量不匹配！"
        echo "   实际用户数: $USER_COUNT，期望: 4"
    fi
else
    echo "❌ 创建失败: $TARGET_CSV"
    exit 1
fi

