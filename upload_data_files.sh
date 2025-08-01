#!/bin/bash

# 数据文件上传脚本
# 使用方法: ./upload_data_files.sh

echo "🚀 开始上传分割的数据文件到GitHub..."

# 检查是否在正确的目录
if [ ! -f "split_large_files.py" ]; then
    echo "❌ 错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 添加所有分割的CSV文件
echo "📁 添加上海主板数据文件..."
git add 上海主板_历史数据_2018至今_20250729_233546_副本2_part_*.csv

echo "📁 添加深圳主板数据文件..."
git add 深圳主板_历史数据_2018至今_20250729_233546_副本_part_*.csv

# 添加合并脚本
echo "📁 添加合并脚本..."
git add merge_*.py

# 添加上传指南
echo "📁 添加上传指南..."
git add 数据上传指南.md

# 提交更改
echo "💾 提交更改..."
git commit -m "添加分割的股票历史数据文件 (上海18个 + 深圳16个 + 合并脚本)"

# 推送到GitHub
echo "🚀 推送到GitHub..."
git push origin main

echo "✅ 上传完成！"
echo ""
echo "📋 上传的文件统计:"
echo "- 上海主板数据: 18个分割文件"
echo "- 深圳主板数据: 16个分割文件" 
echo "- 合并脚本: 2个"
echo "- 上传指南: 1个"
echo ""
echo "📝 用户下载后运行以下命令合并数据:"
echo "python merge_上海主板_历史数据_2018至今_20250729_233546_副本2.py"
echo "python merge_深圳主板_历史数据_2018至今_20250729_233546_副本.py" 