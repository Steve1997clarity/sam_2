class AdvancedSAM2Interface {
    constructor() {
        this.imageCanvas = document.getElementById('imageCanvas');
        this.overlayCanvas = document.getElementById('overlayCanvas');
        this.ctx = this.imageCanvas.getContext('2d');
        
        this.imageData = null;
        this.imageWidth = 0;
        this.imageHeight = 0;
        this.sessionId = null;
        
        // 存储状态
        this.points = [];
        this.currentMask = null; // 当前显示的mask
        this.isAddingPoints = false;
        
        // 模式
        this.currentMode = 'hover'; // 'hover', 'click', 'edit'
        
        // 加载状态管理
        this.imageProcessing = document.getElementById('imageProcessing');
        this.processingText = document.getElementById('processingText');
        this.isProcessing = false;
        
        this.initEventListeners();
        this.initUI();
    }
    
    initEventListeners() {
        const imageInput = document.getElementById('imageInput');
        const uploadBtn = document.getElementById('uploadBtn');
        
        uploadBtn.addEventListener('click', () => {
            imageInput.click();
        });
        
        imageInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.uploadImage(file);
            }
        });
        
        // 工具栏按钮事件
        this.initToolbarEvents();
        
        // 画布事件
        this.initCanvasEvents();
    }
    
    initToolbarEvents() {
        // 模式切换
        document.getElementById('hoverMode').addEventListener('click', () => {
            this.setMode('hover');
        });
        
        document.getElementById('clickMode').addEventListener('click', () => {
            this.setMode('click');
        });
        
        // 操作按钮
        document.getElementById('clearPoints').addEventListener('click', () => {
            this.clearAllPoints();
        });
        
        document.getElementById('undoPoint').addEventListener('click', () => {
            this.undoLastPoint();
        });
        
        document.getElementById('exportMasks').addEventListener('click', () => {
            this.exportSelectedMasks();
        });
        
        // 模型选择器
        document.getElementById('modelSelect').addEventListener('change', (e) => {
            if (e.target.value) {
                this.switchModel(e.target.value);
            }
        });
    }
    
    initCanvasEvents() {
        // 检测是否为移动设备
        this.isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        
        // 统一的事件处理器
        const handlePointerMove = (e) => {
            if (this.currentMode === 'hover') {
                this.handlePointerMove(e);
            }
            this.updateCurrentPosition(e);
        };
        
        const handlePointerClick = (e) => {
            if (this.currentMode === 'click') {
                // 移动端：长按为负向点，普通点击为正向点
                // 桌面端：Shift + 点击 = 负向点，普通点击 = 正向点
                const label = (this.isMobile ? this.isLongPress : e.shiftKey) ? 0 : 1;
                this.handlePointerClick(e, label);
            }
        };
        
        const handlePointerLeave = () => {
            if (this.currentMode === 'hover') {
                this.clearHoverContours();
            }
            this.updateCurrentPosition(null);
        };
        
        // 鼠标事件（桌面端）
        this.imageCanvas.addEventListener('mousemove', handlePointerMove);
        this.imageCanvas.addEventListener('click', handlePointerClick);
        this.imageCanvas.addEventListener('mouseleave', handlePointerLeave);
        
        // 触摸事件（移动端）
        this.imageCanvas.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false });
        this.imageCanvas.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
        this.imageCanvas.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: false });
        
        // 禁用右键菜单
        this.imageCanvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
        });
        
        // 长按检测变量
        this.touchStartTime = 0;
        this.longPressThreshold = 500; // 500ms
        this.isLongPress = false;
        this.touchStartPos = null;
        this.longPressTimer = null;
    }
    
    initUI() {
        // 初始化模式按钮状态
        this.setMode('hover');
        
        // 加载可用模型列表
        this.loadModels();
    }
    
    setMode(mode) {
        this.currentMode = mode;
        
        // 更新按钮状态
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.getElementById(mode + 'Mode').classList.add('active');
        
        // 更新鼠标样式
        switch(mode) {
            case 'hover':
                this.imageCanvas.style.cursor = 'default';
                break;
            case 'click':
                this.imageCanvas.style.cursor = 'crosshair';
                break;
        }
        
        // 清除悬浮效果
        if (mode !== 'hover') {
            this.clearHoverContours();
        }
    }
    
    async uploadImage(file) {
        this.showLoading(true);
        this.updateUploadStatus('正在处理图片...');
        
        try {
            // 第一步：在前端处理图片，确保方向和尺寸正确
            const processedImageData = await this.processImageFile(file);
            
            this.updateUploadStatus('正在上传图片...');
            
            // 第二步：将处理后的图片发送给后端
            const response = await fetch('/upload', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image_data: processedImageData.dataUrl,
                    width: processedImageData.width,
                    height: processedImageData.height
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.imageData = processedImageData.dataUrl;
                this.imageWidth = processedImageData.width;
                this.imageHeight = processedImageData.height;
                this.sessionId = result.session_id;
                
                await this.displayImage();
                this.showImageContainer(true);
                this.updateUploadStatus('图片上传成功！');
                this.updateImageDimensions(`${this.imageWidth} × ${this.imageHeight}`);
                
                // 重置状态
                this.points = [];
                this.currentMask = null;
                this.lastHoverKey = null; // 重置悬浮缓存
                this.updatePointsDisplay();
                this.updateMaskDisplay();
                
            } else {
                this.updateUploadStatus(`错误: ${result.error}`);
            }
        } catch (error) {
            console.error('上传错误:', error);
            this.updateUploadStatus('上传失败，请重试');
        } finally {
            this.showLoading(false);
        }
    }
    
    async processImageFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                const img = new Image();
                
                img.onload = () => {
                    // 创建 canvas 来处理图片
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    
                    // 设置 canvas 尺寸为图片实际尺寸
                    canvas.width = img.naturalWidth;
                    canvas.height = img.naturalHeight;
                    
                    // 绘制图片到 canvas（浏览器会自动处理 EXIF 方向）
                    ctx.drawImage(img, 0, 0);
                    
                    // 获取处理后的图片数据
                    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
                    
                    resolve({
                        dataUrl: dataUrl,
                        width: canvas.width,
                        height: canvas.height
                    });
                };
                
                img.onerror = () => {
                    reject(new Error('图片加载失败'));
                };
                
                img.src = e.target.result;
            };
            
            reader.onerror = () => {
                reject(new Error('文件读取失败'));
            };
            
            reader.readAsDataURL(file);
        });
    }
    
    async displayImage() {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                // 计算显示尺寸（保持宽高比，最大宽度800px）
                const maxWidth = 800;
                const ratio = Math.min(maxWidth / this.imageWidth, maxWidth / this.imageHeight);
                const displayWidth = this.imageWidth * ratio;
                const displayHeight = this.imageHeight * ratio;
                
                // 设置canvas尺寸
                this.imageCanvas.width = displayWidth;
                this.imageCanvas.height = displayHeight;
                this.imageCanvas.style.width = displayWidth + 'px';
                this.imageCanvas.style.height = displayHeight + 'px';
                
                // 设置SVG overlay尺寸
                this.overlayCanvas.setAttribute('width', displayWidth);
                this.overlayCanvas.setAttribute('height', displayHeight);
                this.overlayCanvas.style.width = displayWidth + 'px';
                this.overlayCanvas.style.height = displayHeight + 'px';
                
                // 绘制图片
                this.ctx.clearRect(0, 0, displayWidth, displayHeight);
                this.ctx.drawImage(img, 0, 0, displayWidth, displayHeight);
                
                resolve();
            };
            img.src = this.imageData;
        });
    }
    
    // 统一的坐标获取方法
    getEventCoordinates(e) {
        const rect = this.imageCanvas.getBoundingClientRect();
        let clientX, clientY;
        
        if (e.touches && e.touches.length > 0) {
            // 触摸事件
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else if (e.changedTouches && e.changedTouches.length > 0) {
            // 触摸结束事件
            clientX = e.changedTouches[0].clientX;
            clientY = e.changedTouches[0].clientY;
        } else {
            // 鼠标事件
            clientX = e.clientX;
            clientY = e.clientY;
        }
        
        // 计算相对于canvas的坐标（不使用devicePixelRatio）
        const canvasX = clientX - rect.left;
        const canvasY = clientY - rect.top;
        
        // 转换为图像坐标
        const scaleX = this.imageWidth / rect.width;
        const scaleY = this.imageHeight / rect.height;
        
        const x = Math.round(canvasX * scaleX);
        const y = Math.round(canvasY * scaleY);
        
        // 确保坐标在图像范围内
        const clampedX = Math.max(0, Math.min(this.imageWidth - 1, x));
        const clampedY = Math.max(0, Math.min(this.imageHeight - 1, y));
        
        return { x: clampedX, y: clampedY, clientX, clientY };
    }
    
    // 触摸开始事件
    handleTouchStart(e) {
        e.preventDefault();
        
        if (e.touches.length !== 1) return; // 只处理单点触摸
        
        this.touchStartTime = Date.now();
        this.isLongPress = false;
        
        const coords = this.getEventCoordinates(e);
        this.touchStartPos = coords;
        
        // 开始长按检测
        this.longPressTimer = setTimeout(() => {
            this.isLongPress = true;
            // 可以在这里添加长按反馈，如震动
            if (navigator.vibrate) {
                navigator.vibrate(50);
            }
        }, this.longPressThreshold);
        
        // 如果是悬浮模式，开始显示预览
        if (this.currentMode === 'hover') {
            this.handlePointerMove(e);
        }
    }
    
    // 触摸移动事件
    handleTouchMove(e) {
        e.preventDefault();
        
        if (e.touches.length !== 1) {
            this.clearLongPress();
            return;
        }
        
        const coords = this.getEventCoordinates(e);
        
        // 如果移动距离超过阈值，取消长按
        if (this.touchStartPos) {
            const distance = Math.sqrt(
                Math.pow(coords.clientX - this.touchStartPos.clientX, 2) +
                Math.pow(coords.clientY - this.touchStartPos.clientY, 2)
            );
            
            if (distance > 10) { // 10px阈值
                this.clearLongPress();
            }
        }
        
        // 悬浮模式下显示预览
        if (this.currentMode === 'hover') {
            this.handlePointerMove(e);
        }
        
        this.updateCurrentPosition(e);
    }
    
    // 触摸结束事件
    handleTouchEnd(e) {
        e.preventDefault();
        
        // 清除长按定时器
        this.clearLongPress();
        
        // 如果是点击模式且触摸时间不太长（避免误触）
        if (this.currentMode === 'click') {
            const touchDuration = Date.now() - this.touchStartTime;
            if (touchDuration < 2000) { // 2秒内的触摸视为有效点击
                const label = this.isLongPress ? 0 : 1;
                this.handlePointerClick(e, label);
            }
        }
        
        // 清除悬浮状态
        if (this.currentMode === 'hover') {
            setTimeout(() => {
                this.clearHoverContours();
                this.updateCurrentPosition(null);
            }, 100);
        }
    }
    
    // 清除长按检测
    clearLongPress() {
        if (this.longPressTimer) {
            clearTimeout(this.longPressTimer);
            this.longPressTimer = null;
        }
    }
    
    async handlePointerMove(e) {
        if (this.currentMode !== 'hover' || !this.sessionId) return;
        
        const coords = this.getEventCoordinates(e);
        
        // 网格缓存，避免重复预测相同区域
        const gridSize = 20;
        const gridX = Math.floor(coords.x / gridSize) * gridSize;
        const gridY = Math.floor(coords.y / gridSize) * gridSize;
        const cacheKey = `${gridX},${gridY}`;
        
        if (this.lastHoverKey === cacheKey) {
            return;
        }
        
        this.lastHoverKey = cacheKey;
        
        // 节流处理
        if (this.hoverTimeout) {
            clearTimeout(this.hoverTimeout);
            // 如果上一个请求还在进行中，不要隐藏加载状态
        }
        
        // 清除之前的悬浮效果
        this.clearHoverContours();
        
        this.hoverTimeout = setTimeout(async () => {
            await this.predictHoverMask(coords.x, coords.y);
        }, this.isMobile ? 200 : 150); // 移动端稍微慢一点
    }
    
    async handlePointerClick(e, label) {
        if (!this.sessionId) return;
        
        const coords = this.getEventCoordinates(e);
        
        try {
            // 添加点
            const response = await fetch('/add_point', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ x: coords.x, y: coords.y, label })
            });
            
            const result = await response.json();
            
            if (result.success) {
                // 更新本地点列表
                this.points.push({
                    id: result.point_id,
                    x: coords.x,
                    y: coords.y,
                    label: label
                });
                
                this.updatePointsDisplay();
                
                // 在点击模式下，立即预测
                if (this.currentMode === 'click') {
                    await this.predictMultiMasks();
                }
            }
        } catch (error) {
            console.error('添加点失败:', error);
        }
    }
    
    async predictHoverMask(x, y) {
        try {
            // 显示加载状态（悬浮模式使用脉冲效果）
            this.showImageProcessing('正在分析...', true);
            
            // 使用专门的悬浮预测端点，不修改会话状态
            const response = await fetch('/predict_hover', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ x, y })
            });
            
            const result = await response.json();
            
            // 隐藏加载状态
            this.hideImageProcessing();
            
            if (result.success) {
                // 显示悬浮预览
                this.renderHoverContours(result.contours);
            } else {
                this.clearHoverContours();
            }
        } catch (error) {
            console.error('悬浮预测失败:', error);
            this.hideImageProcessing();
            this.clearHoverContours();
        }
    }
    
    async predictMultiMasks() {
        if (this.points.length === 0) {
            this.clearAllContours();
            this.currentMask = null;
            this.updateMaskDisplay();
            return;
        }
        
        try {
            // 显示加载状态
            this.showImageProcessing('正在生成分割结果...');
            
            const response = await fetch('/predict_multi', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            // 隐藏加载状态
            this.hideImageProcessing();
            
            if (result.success && result.mask) {
                this.currentMask = result.mask;
                this.renderMask();
                this.updateMaskDisplay();
            } else {
                this.currentMask = null;
                this.clearAllContours();
                this.updateMaskDisplay();
            }
        } catch (error) {
            console.error('mask预测失败:', error);
            this.hideImageProcessing();
        }
    }
    
    renderHoverContours(contours) {
        this.clearHoverContours();
        
        if (!contours || contours.length === 0) return;
        
        const scaleX = this.imageCanvas.width / this.imageWidth;
        const scaleY = this.imageCanvas.height / this.imageHeight;
        
        contours.forEach((contour) => {
            if (contour.length < 3) return;
            
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            
            let pathData = '';
            contour.forEach((point, i) => {
                const x = point[0] * scaleX;
                const y = point[1] * scaleY;
                
                if (i === 0) {
                    pathData += `M ${x} ${y}`;
                } else {
                    pathData += ` L ${x} ${y}`;
                }
            });
            pathData += ' Z';
            
            path.setAttribute('d', pathData);
            path.setAttribute('class', 'hover-contour');
            path.setAttribute('fill', 'rgba(0, 123, 255, 0.2)');
            path.setAttribute('stroke', '#007bff');
            path.setAttribute('stroke-width', '2');
            path.setAttribute('stroke-dasharray', '5,5');
            
            this.overlayCanvas.appendChild(path);
        });
    }
    
    renderMask() {
        this.clearAllContours();
        
        // 渲染当前mask
        if (!this.currentMask || !this.currentMask.contours) {
            this.renderPoints();
            return;
        }
        
        const scaleX = this.imageCanvas.width / this.imageWidth;
        const scaleY = this.imageCanvas.height / this.imageHeight;
        
        this.currentMask.contours.forEach((contour) => {
            if (contour.length < 3) return;
            
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            
            let pathData = '';
            contour.forEach((point, i) => {
                const x = point[0] * scaleX;
                const y = point[1] * scaleY;
                
                if (i === 0) {
                    pathData += `M ${x} ${y}`;
                } else {
                    pathData += ` L ${x} ${y}`;
                }
            });
            pathData += ' Z';
            
            path.setAttribute('d', pathData);
            path.setAttribute('class', 'selected-mask-contour');
            
            // 统一使用蓝色系配色
            path.setAttribute('fill', 'rgba(0, 123, 255, 0.3)');
            path.setAttribute('stroke', '#007bff');
            path.setAttribute('stroke-width', '2');
            
            this.overlayCanvas.appendChild(path);
        });
        
        // 渲染点
        this.renderPoints();
    }
    
    renderPoints() {
        // 清除旧的点
        this.overlayCanvas.querySelectorAll('.point-marker').forEach(el => el.remove());
        
        const scaleX = this.imageCanvas.width / this.imageWidth;
        const scaleY = this.imageCanvas.height / this.imageHeight;
        
        this.points.forEach((point) => {
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            
            circle.setAttribute('cx', point.x * scaleX);
            circle.setAttribute('cy', point.y * scaleY);
            circle.setAttribute('r', '6');
            circle.setAttribute('class', 'point-marker');
            circle.setAttribute('data-point-id', point.id);
            
            if (point.label === 1) {
                circle.setAttribute('fill', '#00ff00');
                circle.setAttribute('stroke', '#008000');
            } else {
                circle.setAttribute('fill', '#ff0000');
                circle.setAttribute('stroke', '#800000');
            }
            circle.setAttribute('stroke-width', '2');
            
            // 添加点击事件移除点
            circle.addEventListener('click', (e) => {
                e.stopPropagation();
                this.removePoint(point.id);
            });
            
            this.overlayCanvas.appendChild(circle);
        });
    }
    
    
    async removePoint(pointId) {
        try {
            const response = await fetch('/remove_point', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ point_id: pointId })
            });
            
            const result = await response.json();
            
            if (result.success) {
                // 更新本地点列表
                this.points = this.points.filter(p => p.id !== pointId);
                this.updatePointsDisplay();
                
                // 重新预测
                if (this.currentMode === 'click') {
                    await this.predictMultiMasks();
                }
            }
        } catch (error) {
            console.error('移除点失败:', error);
        }
    }
    
    async clearAllPoints() {
        try {
            // 显示加载状态
            this.showImageProcessing('正在清除...');
            
            const response = await fetch('/clear_points', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            // 隐藏加载状态
            this.hideImageProcessing();
            
            if (result.success) {
                this.points = [];
                this.currentMask = null;
                
                this.clearAllContours();
                this.updatePointsDisplay();
                this.updateMaskDisplay();
            }
        } catch (error) {
            console.error('清除点失败:', error);
            this.hideImageProcessing();
        }
    }
    
    async undoLastPoint() {
        if (this.points.length === 0) return;
        
        const lastPoint = this.points[this.points.length - 1];
        await this.removePoint(lastPoint.id);
    }
    
    async exportSelectedMasks() {
        if (!this.currentMask) {
            alert('没有可导出的mask');
            return;
        }
        
        try {
            // 显示加载状态
            this.showImageProcessing('正在导出...');
            
            const response = await fetch('/export_masks', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            // 隐藏加载状态
            this.hideImageProcessing();
            
            if (result.success) {
                // 创建下载链接（这里简化实现，实际可以生成PNG文件）
                const dataStr = JSON.stringify(this.currentMask.contours);
                const dataBlob = new Blob([dataStr], {type: 'application/json'});
                const url = URL.createObjectURL(dataBlob);
                
                const link = document.createElement('a');
                link.href = url;
                link.download = `sam2_mask.json`;
                link.click();
                
                URL.revokeObjectURL(url);
            }
        } catch (error) {
            console.error('导出失败:', error);
            this.hideImageProcessing();
        }
    }
    
    clearHoverContours() {
        this.overlayCanvas.querySelectorAll('.hover-contour').forEach(el => el.remove());
    }
    
    clearAllContours() {
        while (this.overlayCanvas.firstChild) {
            this.overlayCanvas.removeChild(this.overlayCanvas.firstChild);
        }
    }
    
    updateCurrentPosition(e) {
        if (e) {
            const coords = this.getEventCoordinates(e);
            document.getElementById('currentPosition').textContent = `(${coords.x}, ${coords.y})`;
        } else {
            document.getElementById('currentPosition').textContent = '-';
        }
    }
    
    updatePointsDisplay() {
        const count = this.points.length;
        const positiveCount = this.points.filter(p => p.label === 1).length;
        const negativeCount = this.points.filter(p => p.label === 0).length;
        
        document.getElementById('pointsCount').textContent = `${count} (正: ${positiveCount}, 负: ${negativeCount})`;
    }
    
    updateMaskDisplay() {
        const hasValidMask = this.currentMask && this.currentMask.contours && this.currentMask.contours.length > 0;
        const maskText = hasValidMask ? 
            `1 (Score: ${this.currentMask.score.toFixed(3)})` : 
            '0';
        
        document.getElementById('masksCount').textContent = maskText;
    }
    
    // 图片区域加载状态管理
    showImageProcessing(text = '正在处理...', usePulse = false) {
        if (this.isProcessing) return; // 避免重复显示
        
        this.isProcessing = true;
        this.processingText.textContent = text;
        
        // 根据需要添加脉冲效果
        const spinner = this.imageProcessing.querySelector('.processing-spinner');
        if (usePulse) {
            spinner.classList.add('processing-pulse');
        } else {
            spinner.classList.remove('processing-pulse');
        }
        
        this.imageProcessing.style.display = 'flex';
        
        // 淡入效果
        this.imageProcessing.style.opacity = '0';
        requestAnimationFrame(() => {
            this.imageProcessing.style.transition = 'opacity 0.3s ease';
            this.imageProcessing.style.opacity = '1';
        });
    }
    
    hideImageProcessing() {
        if (!this.isProcessing) return;
        
        this.isProcessing = false;
        
        // 淡出效果
        this.imageProcessing.style.transition = 'opacity 0.3s ease';
        this.imageProcessing.style.opacity = '0';
        
        setTimeout(() => {
            this.imageProcessing.style.display = 'none';
            this.imageProcessing.querySelector('.processing-spinner').classList.remove('processing-pulse');
        }, 300);
    }
    
    // 工具方法
    showLoading(show) {
        document.getElementById('loading').style.display = show ? 'flex' : 'none';
    }
    
    showImageContainer(show) {
        document.getElementById('imageContainer').style.display = show ? 'block' : 'none';
    }
    
    updateUploadStatus(text) {
        document.getElementById('uploadStatus').textContent = text;
    }
    
    updateImageDimensions(dimensions) {
        document.getElementById('imageDimensions').textContent = dimensions;
    }
    
    // 模型管理方法
    async loadModels() {
        try {
            const response = await fetch('/get_models');
            const result = await response.json();
            
            if (result.success) {
                this.populateModelSelect(result.models);
                this.updateCurrentModel(result.current_model, result.models);
            } else {
                console.error('获取模型列表失败:', result.error);
            }
        } catch (error) {
            console.error('加载模型列表失败:', error);
        }
    }
    
    populateModelSelect(models) {
        const select = document.getElementById('modelSelect');
        select.innerHTML = '';
        
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.type;
            option.textContent = model.name;
            option.disabled = !model.available;
            option.selected = model.current;
            
            if (!model.available) {
                option.textContent += ' (未安装)';
            }
            
            select.appendChild(option);
        });
    }
    
    updateCurrentModel(currentModel, models) {
        const currentModelSpan = document.getElementById('currentModel');
        const model = models.find(m => m.type === currentModel);
        if (model) {
            currentModelSpan.textContent = model.name;
        }
    }
    
    async switchModel(modelType) {
        try {
            // 显示全局加载
            this.showLoading(true);
            
            const response = await fetch('/switch_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model_type: modelType })
            });
            
            const result = await response.json();
            
            if (result.success) {
                // 更新当前模型显示
                document.getElementById('currentModel').textContent = result.model_name;
                
                // 清空当前状态，因为切换了模型
                this.clearCurrentSession();
                
                // 显示成功消息
                this.showMessage(result.message, 'success');
            } else {
                // 切换失败，恢复选择器状态
                this.loadModels();
                this.showMessage(result.error, 'error');
            }
        } catch (error) {
            console.error('切换模型失败:', error);
            this.loadModels();
            this.showMessage('模型切换失败，请重试', 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    clearCurrentSession() {
        // 清空所有状态
        this.points = [];
        this.currentMask = null;
        this.sessionId = null;
        
        // 清空画布
        this.clearAllContours();
        
        // 更新显示
        this.updatePointsDisplay();
        this.updateMaskDisplay();
        
        // 隐藏图片容器
        this.showImageContainer(false);
        
        // 重置上传状态
        this.updateUploadStatus('');
        document.getElementById('imageInput').value = '';
    }
    
    showMessage(message, type = 'info') {
        // 简单的消息显示，可以用更好的Toast组件替代
        const messageClass = type === 'error' ? 'error' : 'success';
        
        // 创建临时消息元素
        const messageDiv = document.createElement('div');
        messageDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            background: ${type === 'error' ? '#ff4757' : '#2ed573'};
            color: white;
            border-radius: 6px;
            font-size: 14px;
            z-index: 1001;
            animation: slideIn 0.3s ease;
        `;
        messageDiv.textContent = message;
        
        document.body.appendChild(messageDiv);
        
        // 3秒后自动移除
        setTimeout(() => {
            messageDiv.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                if (messageDiv.parentNode) {
                    document.body.removeChild(messageDiv);
                }
            }, 300);
        }, 3000);
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    const app = new AdvancedSAM2Interface();
    
    // 检查服务器状态
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            console.log('服务器状态:', data);
            if (!data.model_loaded) {
                console.warn('SAM 2模型未加载');
            }
        })
        .catch(error => {
            console.error('无法连接到服务器:', error);
        });
});