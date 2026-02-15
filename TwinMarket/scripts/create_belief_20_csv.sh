#!/bin/bash

# 从 belief_1000.csv 中提取20个用户创建 belief_20.csv
# 需要先运行 create_sys_20_db.sh 来获取选择的用户ID

SOURCE_CSV="util/belief/belief_1000.csv"
TARGET_CSV="util/belief/belief_20.csv"
SOURCE_DB="data/sys_20.db"

echo "=== 从 ${SOURCE_CSV} 创建 ${TARGET_CSV} ==="

# 检查源文件是否存在
if [ ! -f "$SOURCE_CSV" ]; then
    echo "❌ 源文件不存在: $SOURCE_CSV"
    exit 1
fi

# 检查数据库是否存在（用于获取用户ID列表）
if [ ! -f "$SOURCE_DB" ]; then
    echo "⚠️  警告: 数据库 $SOURCE_DB 不存在"
    echo "   将使用 belief_100.csv 作为临时方案（如果存在）"
    if [ -f "util/belief/belief_100.csv" ]; then
        echo "   使用 belief_100.csv 的前20行（包括header）"
        head -21 "util/belief/belief_100.csv" > "$TARGET_CSV"
        echo "✅ 创建完成: $TARGET_CSV（使用 belief_100.csv 的前20个用户）"
        exit 0
    else
        echo "❌ 找不到 belief_100.csv，请先运行 create_sys_20_db.sh"
        exit 1
    fi
fi

# 使用 Python 来确保准确匹配
echo "📋 从数据库获取用户ID列表并匹配..."
python3 << 'PYTHON_SCRIPT'
import csv
import sqlite3
import sys

source_csv = "$SOURCE_CSV"
target_csv = "$TARGET_CSV"
source_db = "$SOURCE_DB"

# 获取数据库中的用户ID
conn = sqlite3.connect(source_db)
cur = conn.cursor()
cur.execute('SELECT user_id FROM Profiles ORDER BY user_id')
db_users = [str(row[0]) for row in cur.fetchall()]
conn.close()

# 读取 belief_1000.csv 并创建字典
with open(source_csv, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    belief_dict = {row[0]: row for row in reader}

# 创建 belief_20.csv
with open(target_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)  # header
    
    found_count = 0
    for user_id in db_users:
        if user_id in belief_dict:
            writer.writerow(belief_dict[user_id])
            found_count += 1
        else:
            print(f'⚠️  警告: 用户 {user_id} 在 belief_1000.csv 中未找到', file=sys.stderr)

print(f'✅ 匹配完成: {found_count}/{len(db_users)} 个用户')
PYTHON_SCRIPT

# 验证结果
USER_COUNT=$(tail -n +2 "$TARGET_CSV" | wc -l)

echo ""
echo "✅ 创建完成！"
echo "   - 目标文件: $TARGET_CSV"
echo "   - 用户数量: $USER_COUNT / 20"

if [ "$USER_COUNT" -eq 20 ]; then
    echo "✅ 验证通过！"
else
    echo "⚠️  警告: 用户数量不匹配！"
fi

