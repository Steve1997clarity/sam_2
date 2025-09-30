#!/bin/bash

# SAM2配置文件下载脚本
# 下载所有必需的YAML配置文件

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
CONFIG_DIR="$PROJECT_DIR/sam2_repo/configs/sam2.1"

# 配置文件URL和名称
CONFIG_BASE_URL="https://raw.githubusercontent.com/facebookresearch/sam2/main/configs/sam2.1"

declare -a CONFIG_FILES=(
    "sam2.1_hiera_t.yaml"
    "sam2.1_hiera_s.yaml"
    "sam2.1_hiera_b%2B.yaml:sam2.1_hiera_b+.yaml"
    "sam2.1_hiera_l.yaml"
)

# 输出函数
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}     SAM2 配置文件下载脚本              ${NC}"
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
    if ping -c 1 raw.githubusercontent.com &> /dev/null; then
        print_success "网络连接正常"
    else
        print_error "无法连接到GitHub服务器，请检查网络连接"
        exit 1
    fi
}

# 检查下载工具
check_download_tool() {
    if command -v wget &> /dev/null; then
        DOWNLOAD_CMD="wget"
        DOWNLOAD_ARGS="--quiet --show-progress"
        print_success "使用 wget 进行下载"
    elif command -v curl &> /dev/null; then
        DOWNLOAD_CMD="curl"
        DOWNLOAD_ARGS="-L --progress-bar"
        print_success "使用 curl 进行下载"
    else
        print_error "未找到 wget 或 curl，请先安装下载工具"
        print_info "Ubuntu/Debian: sudo apt-get install wget"
        print_info "CentOS/RHEL: sudo yum install wget"
        exit 1
    fi
}

# 创建目录结构
create_directories() {
    print_info "创建配置文件目录..."
    mkdir -p "$CONFIG_DIR"
    print_success "目录创建完成: $CONFIG_DIR"
}

# 下载单个配置文件
download_config() {
    local config_spec=$1
    local url_name=""
    local local_name=""
    
    # 检查是否包含重命名信息
    if [[ "$config_spec" == *":"* ]]; then
        url_name=$(echo "$config_spec" | cut -d':' -f1)
        local_name=$(echo "$config_spec" | cut -d':' -f2)
    else
        url_name="$config_spec"
        local_name="$config_spec"
    fi
    
    local url="$CONFIG_BASE_URL/$url_name"
    local filepath="$CONFIG_DIR/$local_name"
    
    print_info "下载: $local_name"
    
    # 检查文件是否已存在
    if [ -f "$filepath" ]; then
        # 检查文件是否为空或太小（可能是错误页面）
        local file_size=$(stat -c%s "$filepath" 2>/dev/null || stat -f%z "$filepath" 2>/dev/null || echo 0)
        if [ $file_size -gt 100 ]; then
            print_warning "文件已存在: $local_name"
            return 0
        else
            print_warning "文件已存在但大小异常，重新下载"
        fi
    fi
    
    # 开始下载
    cd "$CONFIG_DIR"
    
    if [ "$DOWNLOAD_CMD" = "wget" ]; then
        if wget $DOWNLOAD_ARGS -O "$local_name" "$url"; then
            print_success "下载完成: $local_name"
        else
            print_error "下载失败: $local_name"
            return 1
        fi
    else
        if curl $DOWNLOAD_ARGS -o "$local_name" "$url"; then
            print_success "下载完成: $local_name"
        else
            print_error "下载失败: $local_name"
            return 1
        fi
    fi
    
    # 验证文件内容（检查是否真的是YAML文件）
    if head -n 1 "$filepath" | grep -q "model:" || head -n 5 "$filepath" | grep -q "_target_"; then
        print_success "文件验证通过: $local_name"
    else
        print_error "文件内容异常，可能下载了错误页面: $local_name"
        cat "$filepath" | head -3  # 显示文件前几行用于调试
        return 1
    fi
    
    cd "$PROJECT_DIR"
    return 0
}

# 下载所有配置文件
download_all_configs() {
    print_info "开始下载配置文件..."
    echo ""
    
    local success_count=0
    local total_count=${#CONFIG_FILES[@]}
    
    for config_file in "${CONFIG_FILES[@]}"; do
        if download_config "$config_file"; then
            ((success_count++))
        fi
        echo ""  # 添加空行分隔
    done
    
    echo ""
    if [ $success_count -eq $total_count ]; then
        print_success "所有配置文件下载完成！($success_count/$total_count)"
    else
        print_warning "部分配置文件下载完成 ($success_count/$total_count)"
    fi
}

# 显示下载结果
show_results() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}           下载结果                     ${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    if [ -d "$CONFIG_DIR" ] && [ "$(ls -A $CONFIG_DIR 2>/dev/null)" ]; then
        print_success "已下载的配置文件:"
        echo ""
        
        for file in "$CONFIG_DIR"/*.yaml; do
            if [ -f "$file" ]; then
                local filename=$(basename "$file")
                local size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
                echo -e "  ${GREEN}✓${NC} $filename (${size} bytes)"
            fi
        done
        
        echo ""
        print_info "配置文件位置: $CONFIG_DIR"
        print_info "现在可以启动SAM2应用了！"
    else
        print_warning "没有找到已下载的配置文件"
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
    
    # 下载配置文件
    download_all_configs
    
    # 显示结果
    show_results
}

# 运行主函数
main "$@"