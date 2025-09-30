#!/bin/bash

# 配置
SESSION_NAME="sam2_app"
PORT=5985

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}     SAM2 Application Status            ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查 tmux 会话
echo -e "${YELLOW}Tmux Session Status:${NC}"
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo -e "  ${GREEN}✓ Session '$SESSION_NAME' is running${NC}"
    
    # 显示会话信息
    echo -e "  ${BLUE}Session info:${NC}"
    tmux list-sessions | grep $SESSION_NAME
else
    echo -e "  ${RED}✗ Session '$SESSION_NAME' is not running${NC}"
fi

echo ""

# 检查端口
echo -e "${YELLOW}Port Status:${NC}"
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo -e "  ${GREEN}✓ Port $PORT is in use${NC}"
    echo -e "  ${BLUE}Process info:${NC}"
    lsof -i:$PORT | grep LISTEN
else
    echo -e "  ${RED}✗ Port $PORT is not in use${NC}"
fi

echo ""

# 检查应用响应
echo -e "${YELLOW}Application Response:${NC}"
if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT | grep -q "200"; then
    echo -e "  ${GREEN}✓ Application is responding (HTTP 200)${NC}"
else
    echo -e "  ${RED}✗ Application is not responding${NC}"
fi

echo ""

# 显示访问 URL
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    PUBLIC_IP=$(curl -s ifconfig.me)
    echo -e "${YELLOW}Access URLs:${NC}"
    echo -e "  Local:  ${GREEN}http://localhost:$PORT${NC}"
    echo -e "  Public: ${GREEN}http://$PUBLIC_IP:$PORT${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"