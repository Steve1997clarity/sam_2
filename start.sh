#!/bin/bash

# 配置
PROJECT_DIR="/home/ubuntu/software/sam2"
CONDA_ENV="sam"
SESSION_NAME="sam2_app"
PORT=3000

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}     Starting SAM2 Application          ${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查是否已有同名 tmux 会话
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo -e "${YELLOW}⚠ Warning: Session '$SESSION_NAME' already exists${NC}"
    echo -e "${YELLOW}Use 'tmux attach -t $SESSION_NAME' to attach to it${NC}"
    echo -e "${YELLOW}Or run ./stop.sh first to stop the existing session${NC}"
    exit 1
fi

# 检查端口是否被占用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${RED}✗ Error: Port $PORT is already in use${NC}"
    echo -e "${RED}Please stop the existing service or choose a different port${NC}"
    exit 1
fi

# 检查项目目录
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}✗ Error: Project directory not found: $PROJECT_DIR${NC}"
    exit 1
fi

# 检查 app.py 文件
if [ ! -f "$PROJECT_DIR/app.py" ]; then
    echo -e "${RED}✗ Error: app.py not found in $PROJECT_DIR${NC}"
    exit 1
fi

# 创建 tmux 会话并运行应用
echo -e "${GREEN}➤ Creating tmux session: $SESSION_NAME${NC}"
tmux new-session -d -s $SESSION_NAME -c $PROJECT_DIR

# 激活 conda 环境并启动应用
echo -e "${GREEN}➤ Activating conda environment: $CONDA_ENV${NC}"
tmux send-keys -t $SESSION_NAME "source ~/miniconda3/etc/profile.d/conda.sh" C-m
tmux send-keys -t $SESSION_NAME "conda activate $CONDA_ENV" C-m

# 等待环境激活
sleep 2

# 启动 Flask 应用
echo -e "${GREEN}➤ Starting Flask application on port $PORT${NC}"
tmux send-keys -t $SESSION_NAME "cd $PROJECT_DIR" C-m
tmux send-keys -t $SESSION_NAME "python app.py" C-m

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ SAM2 Application started successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo -e "  View logs:    ${GREEN}tmux attach -t $SESSION_NAME${NC}"
echo -e "  Detach:       ${GREEN}Ctrl+B then D${NC} (when attached)"
echo -e "  Stop app:     ${GREEN}./stop.sh${NC}"
echo -e "  Check status: ${GREEN}tmux ls${NC}"
echo ""
echo -e "${YELLOW}Application URL:${NC}"
echo -e "  Local:        ${GREEN}http://localhost:$PORT${NC}"
echo -e "  Public:       ${GREEN}http://$(curl -s ifconfig.me):$PORT${NC}"
echo ""

# 等待几秒检查应用是否成功启动
echo -e "${YELLOW}Waiting for application to start...${NC}"
sleep 5

# 检查应用是否在运行
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${GREEN}✓ Application is running on port $PORT${NC}"
else
    echo -e "${RED}⚠ Warning: Application may not have started correctly${NC}"
    echo -e "${RED}Use 'tmux attach -t $SESSION_NAME' to check logs${NC}"
fi