#!/bin/bash

# 简单的GitHub推送脚本
echo "🚀 准备推送到GitHub..."

# 检查是否提供了仓库URL
if [ $# -eq 0 ]; then
    echo "❌ 请提供GitHub仓库URL"
    echo "使用方法: ./push_to_github.sh <仓库URL>"
    echo "例如: ./push_to_github.sh https://github.com/yamijin/quantitative-trading-strategy.git"
    exit 1
fi

REPO_URL=$1

echo "📦 添加远程仓库: $REPO_URL"
git remote add origin "$REPO_URL"

echo "🚀 推送到GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 推送成功！"
    echo "📊 您的量化交易策略已成功上传到GitHub"
    echo "🔗 访问地址: $REPO_URL"
    echo ""
    echo "📋 下一步建议:"
    echo "1. 在GitHub上完善项目描述"
    echo "2. 添加项目标签: quantitative-trading, machine-learning, python"
    echo "3. 设置GitHub Pages (可选)"
    echo "4. 邀请其他开发者参与"
else
    echo "❌ 推送失败"
    echo "请检查:"
    echo "1. 仓库URL是否正确"
    echo "2. 网络连接是否正常"
    echo "3. GitHub凭据是否正确"
fi 