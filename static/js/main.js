class SAM2Interface {
    constructor() {
        // 画布元素
        this.canvas = document.getElementById('imageCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.overlay = document.getElementById('overlayCanvas');
        
        // 图片数据
        this.imageData = null;
        this.imageWidth = 0;
        this.imageHeight = 0;
        
        // 点收集
        this.points = [];
        this.maxPoints = 5;
        
        // 状态
        this.sessionId = null;
        this.isProcessing = false;
        this.currentModel = 'small'; // 默认模型
        
        this.initEventListeners();
        this.loadModels(); // 加载可用模型列表
    }
    
    initEventListeners() {
        // 上传按钮
        document.getElementById('uploadBtn').addEventListener('click', () => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/*';
            
            // 为移动设备添加 capture 属性以优化拍照体验
            if (/iPhone|iPad|iPod|Android/i.test(navigator.userAgent)) {
                input.setAttribute('capture', 'environment');
            }
            
            input.onchange = (e) => this.uploadImage(e.target.files[0]);
            input.click();
        });
        
        // 工具栏按钮
        document.getElementById('clearBtn').addEventListener('click', () => this.clearPoints());
        document.getElementById('undoBtn').addEventListener('click', () => this.removeLastPoint());
        document.getElementById('segmentBtn').addEventListener('click', () => this.segmentImage());
        
        // 模型选择器
        document.getElementById('modelSelect').addEventListener('change', (e) => {
            if (e.target.value) {
                this.switchModel(e.target.value);
            }
        });
        
        // 画布点击事件
        this.canvas.addEventListener('click', (e) => this.handleClick(e));
        
        // 拖放事件
        this.initDragAndDrop();
    }
    
    async uploadImage(file) {
        if (!file) return;
        
        // 检查文件大小（限制在15MB）
        const maxFileSize = 15 * 1024 * 1024; // 15MB
        if (file.size > maxFileSize) {
            this.updateStatus('图片文件太大，请选择小于15MB的图片');
            return;
        }
        
        console.log('开始上传:', file.name, '大小:', (file.size / 1024 / 1024).toFixed(2) + 'MB');
        
        this.showLoading(true);
        this.updateStatus('正在处理图片...');
        
        try {
            // 前端处理图片
            const processedData = await this.processImageFile(file);
            
            this.updateStatus('正在上传图片...');
            
            // 发送到后端
            const response = await fetch('/upload', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_data: processedData.dataUrl,
                    width: processedData.width,
                    height: processedData.height
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.imageData = result.original_image;
                this.imageWidth = processedData.width;
                this.imageHeight = processedData.height;
                this.sessionId = result.session_id;
                
                this.displayImage();
                this.showImageDisplay(true);
                this.updateStatus('图片上传成功，请在图片上点击选择区域（最多5个点）');
                this.clearPoints();
            } else {
                this.updateStatus(`错误: ${result.error}`);
            }
        } catch (error) {
            console.error('上传错误详情:', error);
            console.error('错误堆栈:', error.stack);
            this.updateStatus('上传失败: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }
    
    async processImageFile(file) {
        console.log('处理文件:', file.name, '大小:', (file.size / 1024 / 1024).toFixed(2) + 'MB', '类型:', file.type);
        
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    
                    // 限制最大尺寸为 2048x2048
                    const maxSize = 2048;
                    let width = img.naturalWidth;
                    let height = img.naturalHeight;
                    
                    console.log('原始图片尺寸:', width, 'x', height);
                    
                    // 如果图片太大，进行缩放
                    if (width > maxSize || height > maxSize) {
                        const ratio = Math.min(maxSize / width, maxSize / height);
                        width = Math.round(width * ratio);
                        height = Math.round(height * ratio);
                        console.log('图片缩放至:', width, 'x', height);
                    }
                    
                    canvas.width = width;
                    canvas.height = height;
                    ctx.drawImage(img, 0, 0, width, height);
                    
                    // 压缩质量调低到0.7
                    const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
                    console.log('处理后图片大小:', (dataUrl.length * 0.75 / 1024 / 1024).toFixed(2) + 'MB');
                    
                    resolve({
                        dataUrl: dataUrl,
                        width: width,
                        height: height
                    });
                };
                img.onerror = (err) => {
                    console.error('图片加载失败:', err);
                    reject(new Error('图片加载失败'));
                };
                img.src = e.target.result;
            };
            reader.onerror = (err) => {
                console.error('文件读取失败:', err);
                reject(new Error('文件读取失败'));
            };
            reader.readAsDataURL(file);
        });
    }
    
    displayImage() {
        const img = new Image();
        img.onload = () => {
            // 根据屏幕大小动态调整最大尺寸
            const isMobile = window.innerWidth <= 768;
            const maxWidth = isMobile ? Math.min(window.innerWidth - 40, 600) : 800;
            const maxHeight = isMobile ? Math.min(window.innerHeight - 200, 450) : 600;
            
            // 正确计算缩放比例，保持宽高比
            const widthRatio = maxWidth / this.imageWidth;
            const heightRatio = maxHeight / this.imageHeight;
            const ratio = Math.min(widthRatio, heightRatio, 1); // 不放大图片
            
            const displayWidth = Math.round(this.imageWidth * ratio);
            const displayHeight = Math.round(this.imageHeight * ratio);
            
            console.log('显示尺寸:', displayWidth, 'x', displayHeight, '缩放比例:', ratio.toFixed(3));
            
            // 设置canvas尺寸
            this.canvas.width = displayWidth;
            this.canvas.height = displayHeight;
            this.canvas.style.width = displayWidth + 'px';
            this.canvas.style.height = displayHeight + 'px';
            
            // 设置overlay尺寸
            this.overlay.setAttribute('width', displayWidth);
            this.overlay.setAttribute('height', displayHeight);
            this.overlay.style.width = displayWidth + 'px';
            this.overlay.style.height = displayHeight + 'px';
            
            // 绘制图片
            this.ctx.clearRect(0, 0, displayWidth, displayHeight);
            this.ctx.drawImage(img, 0, 0, displayWidth, displayHeight);
        };
        img.src = this.imageData;
    }
    
    handleClick(e) {
        if (this.isProcessing || this.points.length >= this.maxPoints) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // 转换为原图坐标
        const scaleX = this.imageWidth / rect.width;
        const scaleY = this.imageHeight / rect.height;
        
        const point = {
            x: Math.round(x * scaleX),
            y: Math.round(y * scaleY),
            displayX: x,
            displayY: y
        };
        
        this.points.push(point);
        this.renderPoints();
        this.updatePointsCount();
    }
    
    renderPoints() {
        // 清除旧的点
        this.overlay.innerHTML = '';
        
        this.points.forEach((point, index) => {
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', point.displayX);
            circle.setAttribute('cy', point.displayY);
            circle.setAttribute('r', '8');
            circle.setAttribute('fill', '#00ff00');
            circle.setAttribute('stroke', '#008000');
            circle.setAttribute('stroke-width', '3');
            circle.setAttribute('cursor', 'pointer');
            
            // 点击删除
            circle.addEventListener('click', (e) => {
                e.stopPropagation();
                this.removePoint(index);
            });
            
            this.overlay.appendChild(circle);
        });
    }
    
    removePoint(index) {
        this.points.splice(index, 1);
        this.renderPoints();
        this.updatePointsCount();
    }
    
    clearPoints() {
        this.points = [];
        this.renderPoints();
        this.updatePointsCount();
    }
    
    removeLastPoint() {
        if (this.points.length > 0) {
            this.points.pop();
            this.renderPoints();
            this.updatePointsCount();
        }
    }
    
    async segmentImage() {
        if (this.points.length === 0) {
            alert('请先选择一些点');
            return;
        }
        
        this.isProcessing = true;
        this.showLoading(true);
        this.updateStatus('正在进行分割...');
        
        try {
            const response = await fetch('/segment_with_points', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    points: this.points.map(p => ({ x: p.x, y: p.y, label: 1 }))
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                // 显示分割结果
                this.displayResult(result.result_image);
                this.updateStatus(`分割完成！基于${result.points_count}个点生成了分割结果`);
            } else {
                this.updateStatus(`分割失败: ${result.error}`);
            }
        } catch (error) {
            console.error('分割错误:', error);
            this.updateStatus('分割失败，请重试');
        } finally {
            this.isProcessing = false;
            this.showLoading(false);
        }
    }
    
    displayResult(resultImageData) {
        const img = new Image();
        img.onload = () => {
            // 清除点的显示
            this.overlay.innerHTML = '';
            
            // 绘制结果图片
            const displayWidth = this.canvas.width;
            const displayHeight = this.canvas.height;
            this.ctx.clearRect(0, 0, displayWidth, displayHeight);
            this.ctx.drawImage(img, 0, 0, displayWidth, displayHeight);
            
            // 添加重新选择按钮
            this.showRestartButton();
        };
        img.src = resultImageData;
    }
    
    showRestartButton() {
        let restartBtn = document.getElementById('restartBtn');
        if (!restartBtn) {
            restartBtn = document.createElement('button');
            restartBtn.id = 'restartBtn';
            restartBtn.textContent = '重新选择';
            restartBtn.className = 'btn';
            restartBtn.addEventListener('click', () => this.restart());
            
            const toolbar = document.querySelector('.toolbar');
            toolbar.appendChild(restartBtn);
        }
        restartBtn.style.display = 'inline-block';
    }
    
    restart() {
        // 重新显示原图
        this.displayImage();
        
        // 清除点
        this.clearPoints();
        
        // 隐藏重新选择按钮
        const restartBtn = document.getElementById('restartBtn');
        if (restartBtn) {
            restartBtn.style.display = 'none';
        }
        
        this.updateStatus('请在图片上点击选择区域（最多5个点）');
    }
    
    updateStatus(text) {
        document.getElementById('status').textContent = text;
    }
    
    updatePointsCount() {
        document.getElementById('pointsCount').textContent = `${this.points.length}/${this.maxPoints} 个点`;
    }
    
    showLoading(show) {
        document.getElementById('loading').style.display = show ? 'flex' : 'none';
    }
    
    showImageDisplay(show) {
        document.getElementById('uploadZone').style.display = show ? 'none' : 'block';
        document.getElementById('imageDisplay').style.display = show ? 'block' : 'none';
        document.getElementById('controlsCard').style.display = show ? 'block' : 'none';
    }
    
    async loadModels() {
        try {
            const response = await fetch('/get_models');
            const result = await response.json();
            
            if (result.success) {
                const select = document.getElementById('modelSelect');
                select.innerHTML = '';
                
                result.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.type;
                    option.textContent = model.name;
                    option.disabled = !model.available;
                    
                    if (model.current) {
                        option.selected = true;
                        this.currentModel = model.type;
                    }
                    
                    if (!model.available) {
                        option.textContent += ' (未安装)';
                    }
                    
                    select.appendChild(option);
                });
                
                console.log('当前模型:', this.currentModel);
            }
        } catch (error) {
            console.error('加载模型列表失败:', error);
            const select = document.getElementById('modelSelect');
            select.innerHTML = '<option value="">加载失败</option>';
        }
    }
    
    async switchModel(modelType) {
        if (modelType === this.currentModel) return;
        
        this.showLoading(true);
        this.updateStatus('正在切换模型...');
        
        try {
            const response = await fetch('/switch_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_type: modelType })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.currentModel = modelType;
                
                // 清除当前的点和分割结果，但保留图片
                this.clearPoints();
                
                // 如果保留了会话，显示相应信息
                if (result.sessions_preserved > 0) {
                    this.updateStatus(`已切换到 ${result.model_name}，图片已保留，可以重新选择点进行分割`);
                } else {
                    this.updateStatus(`已切换到 ${result.model_name}`);
                }
                
                console.log('切换到模型:', modelType, '保留会话数:', result.sessions_preserved);
            } else {
                this.updateStatus(`切换失败: ${result.error}`);
                // 恢复选择器
                document.getElementById('modelSelect').value = this.currentModel;
            }
        } catch (error) {
            console.error('模型切换失败:', error);
            this.updateStatus('模型切换失败');
            document.getElementById('modelSelect').value = this.currentModel;
        } finally {
            this.showLoading(false);
        }
    }
    
    initDragAndDrop() {
        const uploadZone = document.getElementById('uploadZone');
        
        // 阻止默认拖放行为
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });
        
        // 拖放视觉反馈
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadZone.addEventListener(eventName, () => {
                uploadZone.classList.add('dragover');
            });
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, () => {
                uploadZone.classList.remove('dragover');
            });
        });
        
        // 处理文件拖放
        uploadZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                // 检查文件类型
                if (file.type.startsWith('image/')) {
                    this.uploadImage(file);
                } else {
                    this.updateStatus('请拖放图片文件（JPG, PNG等格式）');
                }
            }
        });
        
        // 点击上传区域也能选择文件
        uploadZone.addEventListener('click', () => {
            document.getElementById('uploadBtn').click();
        });
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new SAM2Interface();
});