#!/bin/bash

# SAM2模型一键下载脚本
# 适用于云服务器环境

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目路径（自动检测脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
MODELS_DIR="$PROJECT_DIR/models"

# 模型配置
declare -A MODELS=(
    ["tiny"]="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt 39"
    ["small"]="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt 155" 
    ["base_plus"]="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt 323"
    ["large"]="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt 897"
)

declare -A MODEL_NAMES=(
    ["tiny"]="Tiny (最快，39MB)"
    ["small"]="Small (平衡，155MB)"
    ["base_plus"]="Base Plus (高精度，323MB)"
    ["large"]="Large (最高精度，897MB)"
)

# 输出函数
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}     SAM2 模型一键下载脚本             ${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# 检查网络连接
check_network() {
    print_info "检查网络连接..."
    if ping -c 1 dl.fbaipublicfiles.com &> /dev/null; then
        print_success "网络连接正常"
    else
        print_error "无法连接到下载服务器，请检查网络连接"
        exit 1
    fi
}

# 检查磁盘空间
check_disk_space() {
    local needed_space=$1
    local available_space=$(df "$PROJECT_DIR" | awk 'NR==2 {print $4}')
    local available_mb=$((available_space / 1024))
    
    print_info "检查磁盘空间..."
    print_info "可用空间: ${available_mb}MB"
    print_info "需要空间: ${needed_space}MB"
    
    if [ $available_mb -lt $needed_space ]; then
        print_error "磁盘空间不足！需要 ${needed_space}MB，可用 ${available_mb}MB"
        return 1
    else
        print_success "磁盘空间充足"
        return 0
    fi
}

# 检查下载工具
check_download_tool() {
    if command -v wget &> /dev/null; then
        DOWNLOAD_CMD="wget"
        DOWNLOAD_ARGS="--progress=bar:force --continue"
        print_success "使用 wget 进行下载"
    elif command -v curl &> /dev/null; then
        DOWNLOAD_CMD="curl"
        DOWNLOAD_ARGS="-L -C - --progress-bar"
        print_success "使用 curl 进行下载"
    else
        print_error "未找到 wget 或 curl，请先安装下载工具"
        print_info "Ubuntu/Debian: sudo apt-get install wget"
        print_info "CentOS/RHEL: sudo yum install wget"
        exit 1
    fi
}

# 创建目录
create_directories() {
    print_info "创建必要目录..."
    mkdir -p "$MODELS_DIR"
    print_success "目录创建完成: $MODELS_DIR"
}

# 下载单个模型
download_model() {
    local model_key=$1
    local model_info=(${MODELS[$model_key]})
    local url=${model_info[0]}
    local size=${model_info[1]}
    local filename=$(basename "$url")
    local filepath="$MODELS_DIR/$filename"
    
    echo ""
    print_info "准备下载: ${MODEL_NAMES[$model_key]}"
    
    # 检查文件是否已存在
    if [ -f "$filepath" ]; then
        local current_size=$(stat -f%z "$filepath" 2>/dev/null || stat -c%s "$filepath" 2>/dev/null || echo 0)
        local current_size_mb=$((current_size / 1024 / 1024))
        
        if [ $current_size_mb -ge $((size - 5)) ]; then
            print_warning "文件已存在且大小正确: $filename (${current_size_mb}MB)"
            return 0
        else
            print_warning "文件存在但大小不正确，将重新下载"
        fi
    fi
    
    # 检查磁盘空间
    if ! check_disk_space $size; then
        return 1
    fi
    
    # 开始下载
    print_info "开始下载: $filename"
    cd "$MODELS_DIR"
    
    if [ "$DOWNLOAD_CMD" = "wget" ]; then
        if wget $DOWNLOAD_ARGS -O "$filename" "$url"; then
            print_success "下载完成: $filename"
        else
            print_error "下载失败: $filename"
            return 1
        fi
    else
        if curl $DOWNLOAD_ARGS -o "$filename" "$url"; then
            print_success "下载完成: $filename"
        else
            print_error "下载失败: $filename"
            return 1
        fi
    fi
    
    # 验证文件大小
    local downloaded_size=$(stat -f%z "$filepath" 2>/dev/null || stat -c%s "$filepath" 2>/dev/null || echo 0)
    local downloaded_size_mb=$((downloaded_size / 1024 / 1024))
    
    if [ $downloaded_size_mb -ge $((size - 5)) ]; then
        print_success "文件验证通过: ${downloaded_size_mb}MB"
    else
        print_error "文件大小异常: 期望 ${size}MB，实际 ${downloaded_size_mb}MB"
        return 1
    fi
    
    cd "$PROJECT_DIR"
    return 0
}

# 显示菜单
show_menu() {
    echo -e "${YELLOW}请选择要下载的模型:${NC}"
    echo ""
    echo "1) ${MODEL_NAMES[tiny]}"
    echo "2) ${MODEL_NAMES[small]}"
    echo "3) ${MODEL_NAMES[base_plus]}"
    echo "4) ${MODEL_NAMES[large]}"
    echo "5) 下载全部模型 (总计约 1.4GB)"
    echo "6) 推荐组合 (Tiny + Small, 194MB)"
    echo "0) 退出"
    echo ""
}

# 下载推荐组合
download_recommended() {
    local total_size=194
    if ! check_disk_space $total_size; then
        return 1
    fi
    
    print_info "下载推荐组合: Tiny + Small"
    
    if download_model "tiny" && download_model "small"; then
        print_success "推荐组合下载完成！"
        return 0
    else
        print_error "推荐组合下载失败"
        return 1
    fi
}

# 下载全部模型
download_all() {
    local total_size=1414  # 39+155+323+897
    if ! check_disk_space $total_size; then
        return 1
    fi
    
    print_info "开始下载全部模型..."
    
    local success_count=0
    for model_key in tiny small base_plus large; do
        if download_model "$model_key"; then
            ((success_count++))
        fi
    done
    
    if [ $success_count -eq 4 ]; then
        print_success "全部模型下载完成！"
    else
        print_warning "部分模型下载完成 ($success_count/4)"
    fi
}

# 显示下载结果
show_results() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}           下载结果                     ${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    if [ -d "$MODELS_DIR" ] && [ "$(ls -A $MODELS_DIR 2>/dev/null)" ]; then
        print_success "已下载的模型文件:"
        echo ""
        
        for file in "$MODELS_DIR"/*.pt; do
            if [ -f "$file" ]; then
                local filename=$(basename "$file")
                local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
                local size_mb=$((size / 1024 / 1024))
                echo -e "  ${GREEN}✓${NC} $filename (${size_mb}MB)"
            fi
        done
        
        echo ""
        print_info "模型文件位置: $MODELS_DIR"
        print_info "现在可以启动应用并在界面中切换模型了！"
    else
        print_warning "没有找到已下载的模型文件"
    fi
    
    echo ""
}

# 主函数
main() {
    print_header
    
    # 前置检查
    check_network
    check_download_tool
    create_directories
    
    # 显示菜单并处理用户选择
    while true; do
        show_menu
        read -p "请输入选项 (0-6): " choice
        
        case $choice in
            1)
                download_model "tiny"
                ;;
            2)
                download_model "small"
                ;;
            3)
                download_model "base_plus"
                ;;
            4)
                download_model "large"
                ;;
            5)
                download_all
                ;;
            6)
                download_recommended
                ;;
            0)
                print_info "退出下载脚本"
                break
                ;;
            *)
                print_error "无效选项，请重新选择"
                continue
                ;;
        esac
        
        echo ""
        read -p "按回车键继续或输入 'q' 退出: " continue_choice
        if [ "$continue_choice" = "q" ]; then
            break
        fi
    done
    
    # 显示结果
    show_results
}

# 运行主函数
main "$@"