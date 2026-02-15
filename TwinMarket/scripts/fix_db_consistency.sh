#!/bin/bash

# 修复数据库一致性问题：为缺失策略记录的用户添加默认策略

DB_PATH="${1:-data/sys_100.db}"
DEFAULT_STRATEGY="${2:-技术面}"

if [ ! -f "$DB_PATH" ]; then
    echo "❌ 数据库文件不存在: $DB_PATH"
    echo "用法: $0 [数据库路径] [默认策略]"
    echo "示例: $0 data/sys_100.db 技术面"
    exit 1
fi

echo "=== 修复数据库一致性: $DB_PATH ==="
echo "默认策略: $DEFAULT_STRATEGY"
echo ""

# 备份数据库
BACKUP_PATH="${DB_PATH}.backup.$(date +%Y%m%d_%H%M%S)"
echo "📦 创建数据库备份: $BACKUP_PATH"
cp "$DB_PATH" "$BACKUP_PATH"

# 找出缺失策略记录的用户并添加默认策略
echo "🔧 修复缺失的策略记录..."

sqlite3 "$DB_PATH" <<EOF
-- 为缺失策略记录的用户添加默认策略
INSERT INTO Strategy (user_id, strategy)
SELECT p.user_id, '$DEFAULT_STRATEGY' as strategy
FROM Profiles p 
LEFT JOIN Strategy s ON p.user_id = s.user_id 
WHERE s.user_id IS NULL;

-- 显示修复结果
SELECT '修复完成，新增策略记录数: ' || changes();
EOF

# 验证修复结果
echo ""
echo "✅ 验证修复结果..."
MISSING_COUNT=$(sqlite3 "$DB_PATH" "
    SELECT COUNT(*) 
    FROM Profiles p 
    LEFT JOIN Strategy s ON p.user_id = s.user_id 
    WHERE s.user_id IS NULL;
" 2>/dev/null)

if [ "$MISSING_COUNT" -eq 0 ]; then
    echo "✅ 所有用户现在都有策略记录了！"
    echo "📦 备份文件: $BACKUP_PATH"
else
    echo "❌ 仍有 $MISSING_COUNT 个用户缺少策略记录"
    echo "⚠️  请检查数据库或手动修复"
fi

