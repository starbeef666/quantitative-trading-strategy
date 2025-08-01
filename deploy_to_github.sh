#!/bin/bash

# A股量化交易策略 - GitHub部署脚本
# 使用方法: ./deploy_to_github.sh

set -e  # 遇到错误立即退出

echo "🚀 开始部署A股量化交易策略到GitHub..."
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 检查Git是否安装
if ! command -v git &> /dev/null; then
    print_error "Git未安装，请先安装Git"
    echo "安装命令:"
    echo "  macOS: brew install git"
    echo "  Ubuntu: sudo apt-get install git"
    echo "  Windows: 下载 https://git-scm.com/"
    exit 1
fi

print_success "Git已安装: $(git --version)"

# 检查Git LFS是否安装
if ! command -v git-lfs &> /dev/null; then
    print_warning "Git LFS未安装，建议安装以管理大文件"
    echo "安装命令:"
    echo "  macOS: brew install git-lfs"
    echo "  Ubuntu: sudo apt-get install git-lfs"
    echo "  Windows: 下载 https://git-lfs.github.com/"
    echo ""
    echo "是否继续？(y/n)"
    read -p "选择: " continue_without_lfs
    if [[ $continue_without_lfs != "y" && $continue_without_lfs != "Y" ]]; then
        print_error "用户取消操作"
        exit 1
    fi
else
    print_success "Git LFS已安装: $(git-lfs --version)"
fi

# 检查当前目录
current_dir=$(pwd)
print_info "当前工作目录: $current_dir"

# 检查是否在正确的项目目录
if [[ ! -f "README.md" ]] || [[ ! -f "prd" ]]; then
    print_error "当前目录不是项目根目录，请确保在包含README.md和prd的目录中运行此脚本"
    exit 1
fi

print_success "项目目录检查通过"

# 初始化Git仓库（如果还没有）
if [ ! -d ".git" ]; then
    print_info "初始化Git仓库..."
    git init
    print_success "Git仓库初始化完成"
else
    print_info "Git仓库已存在"
fi

# 设置Git LFS（如果已安装）
if command -v git-lfs &> /dev/null; then
    print_info "设置Git LFS..."
    git lfs install
    
    # 跟踪大文件
    git lfs track "*.csv"
    git lfs track "*.pkl"
    git lfs track "*.h5"
    git lfs track "*.parquet"
    git lfs track "*.hdf5"
    git lfs track "*.xlsx"
    git lfs track "*.xls"
    git lfs track "*.model"
    git lfs track "*.joblib"
    git lfs track "*.pickle"
    
    print_success "Git LFS配置完成"
else
    print_warning "跳过Git LFS配置（未安装）"
fi

# 检查Git配置
if [[ -z "$(git config user.name)" ]] || [[ -z "$(git config user.email)" ]]; then
    print_warning "Git用户信息未配置"
    echo "请配置Git用户信息:"
    read -p "请输入您的GitHub用户名: " github_username
    read -p "请输入您的邮箱: " github_email
    
    if [[ -n "$github_username" && -n "$github_email" ]]; then
        git config user.name "$github_username"
        git config user.email "$github_email"
        print_success "Git用户信息配置完成"
    else
        print_error "用户信息不完整，请手动配置"
        echo "手动配置命令:"
        echo "  git config user.name '您的用户名'"
        echo "  git config user.email '您的邮箱'"
        exit 1
    fi
else
    print_success "Git用户信息已配置"
fi

# 添加所有文件
print_info "添加文件到Git..."
git add .

# 检查是否有文件需要提交
if git diff --cached --quiet; then
    print_warning "没有文件需要提交"
    echo "可能的原因:"
    echo "1. 所有文件已经被提交"
    echo "2. 文件被.gitignore忽略"
    echo "3. 没有新文件"
    
    echo ""
    echo "是否查看当前状态？(y/n)"
    read -p "选择: " show_status
    if [[ $show_status == "y" || $show_status == "Y" ]]; then
        echo ""
        echo "当前Git状态:"
        git status
        echo ""
        echo "被忽略的文件:"
        git status --ignored
    fi
else
    print_success "文件添加完成"
fi

# 提交更改
print_info "提交更改..."
commit_message="🎯 初始化A股量化交易策略项目

🔥 核心功能:
- V28.0 AI增强策略，期望收益1.71%
- Top 10 AI特征重要性分析
- 完整的策略文档和实现指南
- 回测引擎和结果分析

📊 重要发现:
- ma5_trend是最重要特征(15.44%)
- close_vs_low_4d是第二重要特征(10.58%)
- 成交量相关特征占据重要地位

🔧 技术栈:
- Python + pandas + numpy
- LightGBM机器学习
- 传统技术分析 + AI评分

📚 文档:
- 详细的README.md
- 特征分析报告
- 项目结构说明
- 贡献指南

⚠️ 免责声明: 仅供学习研究，不构成投资建议"

git commit -m "$commit_message"
print_success "本地提交完成"

# 询问用户是否要推送到GitHub
echo ""
echo "=========================================="
print_info "本地提交完成！"
echo ""
echo "🤔 是否要推送到GitHub？"
echo "1. 是，推送到GitHub"
echo "2. 否，仅本地提交"
echo "3. 查看当前状态"
read -p "请选择 (1/2/3): " choice

case $choice in
    1)
        echo ""
        print_info "推送到GitHub..."
        echo "请确保您已经："
        echo "1. 在GitHub上创建了仓库"
        echo "2. 配置了GitHub凭据"
        echo ""
        
        read -p "请输入GitHub仓库URL (例如: https://github.com/用户名/仓库名.git): " repo_url
        
        if [ -n "$repo_url" ]; then
            # 检查远程仓库是否已存在
            if git remote get-url origin &> /dev/null; then
                print_info "更新远程仓库地址..."
                git remote set-url origin "$repo_url"
            else
                print_info "添加远程仓库..."
                git remote add origin "$repo_url"
            fi
            
            # 推送到GitHub
            print_info "推送到GitHub..."
            if git push -u origin main; then
                print_success "推送成功！"
                echo ""
                echo "🎉 部署成功！"
                echo "📊 您的量化交易策略已成功上传到GitHub"
                echo "🔗 访问地址: $repo_url"
                echo ""
                echo "📋 下一步建议:"
                echo "1. 在GitHub上完善项目描述"
                echo "2. 添加项目标签: quantitative-trading, machine-learning, python"
                echo "3. 设置GitHub Pages (可选)"
                echo "4. 邀请其他开发者参与"
                
                # 如果使用Git LFS，推送LFS文件
                if command -v git-lfs &> /dev/null; then
                    echo ""
                    print_info "推送Git LFS文件..."
                    if git lfs push --all origin main; then
                        print_success "Git LFS文件推送完成"
                    else
                        print_warning "Git LFS文件推送失败，请手动执行: git lfs push --all origin main"
                    fi
                fi
            else
                print_error "推送失败"
                echo ""
                echo "可能的解决方案:"
                echo "1. 检查网络连接"
                echo "2. 确认GitHub凭据正确"
                echo "3. 确认仓库URL正确"
                echo "4. 如果使用Token，确保有推送权限"
                echo ""
                echo "手动推送命令:"
                echo "  git push -u origin main"
            fi
        else
            print_error "未提供仓库URL，跳过推送"
        fi
        ;;
    2)
        print_success "仅完成本地提交"
        echo ""
        echo "💡 提示: 稍后可以使用以下命令推送到GitHub:"
        echo "   git remote add origin <仓库URL>"
        echo "   git push -u origin main"
        ;;
    3)
        echo ""
        echo "当前Git状态:"
        git status
        echo ""
        echo "最近提交:"
        git log --oneline -5
        ;;
    *)
        print_error "无效选择"
        ;;
esac

echo ""
echo "=========================================="
print_success "部署脚本执行完成！"
echo "📚 更多信息请查看README.md文件"
echo "🔧 如需帮助，请查看GitHub上传指南.md文件" 