#!/bin/bash

# 配置
SESSION_NAME="sam2_app"
PORT=3000

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}     Stopping SAM2 Application          ${NC}"
echo -e "${YELLOW}========================================${NC}"

# 检查 tmux 会话是否存在
if ! tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo -e "${YELLOW}⚠ No running session found: $SESSION_NAME${NC}"
    
    # 检查是否有进程在使用端口
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${YELLOW}But port $PORT is still in use. Attempting to kill the process...${NC}"
        
        # 获取占用端口的进程 PID
        PID=$(lsof -t -i:$PORT)
        if [ ! -z "$PID" ]; then
            echo -e "${YELLOW}Killing process $PID using port $PORT${NC}"
            kill -9 $PID
            sleep 2
            echo -e "${GREEN}✓ Process killed${NC}"
        fi
    fi
else
    # 发送 Ctrl+C 到 tmux 会话以优雅关闭 Flask
    echo -e "${YELLOW}➤ Sending interrupt signal to Flask application${NC}"
    tmux send-keys -t $SESSION_NAME C-c
    sleep 2
    
    # 发送 exit 命令
    echo -e "${YELLOW}➤ Closing tmux session${NC}"
    tmux send-keys -t $SESSION_NAME "exit" C-m
    sleep 1
    
    # 强制结束会话（如果还存在）
    if tmux has-session -t $SESSION_NAME 2>/dev/null; then
        echo -e "${YELLOW}➤ Force killing tmux session${NC}"
        tmux kill-session -t $SESSION_NAME
    fi
    
    echo -e "${GREEN}✓ Tmux session '$SESSION_NAME' stopped${NC}"
fi

# 最终检查
sleep 1
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${RED}⚠ Warning: Port $PORT is still in use${NC}"
    echo -e "${RED}You may need to manually kill the process${NC}"
    echo -e "${YELLOW}Run: ${NC}lsof -i:$PORT${NC} to see what's using it"
else
    echo -e "${GREEN}✓ Port $PORT is free${NC}"
fi

# 检查所有 tmux 会话
echo ""
echo -e "${YELLOW}Current tmux sessions:${NC}"
tmux ls 2>/dev/null || echo -e "${GREEN}No active tmux sessions${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ SAM2 Application stopped successfully!${NC}"
echo -e "${GREEN}========================================${NC}"