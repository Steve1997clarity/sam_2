import os
import json
import sys
from flask import Flask, request, jsonify, render_template, session
import torch
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO
import uuid

sys.path.append('./sam2_repo')

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'sam2_interactive_segmentation_key'

# 全局变量
predictor = None
mask_generator = None
current_model = 'small'  # 默认使用small模型

# 模型配置 - 使用初版的路径结构
MODELS_CONFIG = {
    'tiny': {
        'checkpoint': 'models/sam2.1_hiera_tiny.pt',
        'config': 'configs/sam2.1/sam2.1_hiera_t.yaml',
        'name': 'Tiny (最快)'
    },
    'small': {
        'checkpoint': 'models/sam2.1_hiera_small.pt',
        'config': 'configs/sam2.1/sam2.1_hiera_s.yaml',
        'name': 'Small (平衡)'
    },
    'base_plus': {
        'checkpoint': 'models/sam2.1_hiera_base_plus.pt',
        'config': 'configs/sam2.1/sam2.1_hiera_b+.yaml',
        'name': 'Base Plus (高精度)'
    },
    'large': {
        'checkpoint': 'models/sam2.1_hiera_large.pt',
        'config': 'configs/sam2.1/sam2.1_hiera_l.yaml',
        'name': 'Large (最高精度)'
    }
}

# 存储每个会话的状态
sessions = {}

def init_sam2_model(model_type='small'):
    """初始化SAM 2模型"""
    global predictor, mask_generator, current_model
    try:
        if model_type not in MODELS_CONFIG:
            print(f"不支持的模型类型: {model_type}")
            return False
            
        model_config = MODELS_CONFIG[model_type]
        sam2_checkpoint = model_config['checkpoint']
        model_cfg = model_config['config']
        
        # 检查模型文件是否存在
        if not os.path.exists(sam2_checkpoint):
            print(f"模型文件不存在: {sam2_checkpoint}")
            return False
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"加载模型: {model_config['name']}")
        print(f"使用设备: {device}")
        
        # 使用初版的简单方式
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        
        # 创建预测器和自动mask生成器
        predictor = SAM2ImagePredictor(sam2_model)
        mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
        
        current_model = model_type
        print(f"SAM 2模型加载成功: {model_config['name']}")
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False

def get_session_id():
    """获取或创建会话ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def init_session(session_id):
    """初始化会话状态"""
    if session_id not in sessions:
        sessions[session_id] = {
            'image': None,
            'image_path': None,
            'points': [],  # 存储所有点击点 [{'x': x, 'y': y, 'label': 1/-1, 'id': uuid}]
            'masks': [],   # 存储所有mask结果
            'selected_masks': set(),  # 选中的mask ID集合
            'mask_segmentations': []  # 存储原始分割结果用于点击检测
        }

def extract_contours(mask):
    """从掩码中提取轮廓"""
    try:
        # 确保mask是numpy数组且为uint8类型
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        mask = (mask * 255).astype(np.uint8)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 转换轮廓为前端可用的格式
        contour_points = []
        for contour in contours:
            if len(contour) > 10:  # 过滤太小的轮廓
                points = contour.reshape(-1, 2).tolist()
                contour_points.append(points)
        
        return contour_points
    except Exception as e:
        print(f"轮廓提取错误: {e}")
        return []

def generate_mask_overlay(image_np, masks, mask_segmentations, selected_ids):
    """生成带有mask的渲染图"""
    try:
        # 创建叠加图
        overlay = image_np.copy().astype(np.float32)
        
        # 定义颜色表
        colors = [
            [255, 0, 0],    # 红色
            [0, 255, 0],    # 绿色
            [0, 0, 255],    # 蓝色
            [255, 255, 0],  # 黄色
            [255, 0, 255],  # 紫色
            [0, 255, 255],  # 青色
            [255, 128, 0],  # 橙色
            [128, 0, 255],  # 紫蓝色
            [0, 128, 255],  # 天蓝色
            [255, 0, 128],  # 粉色
        ]
        
        for i, mask_info in enumerate(masks):
            mask_id = mask_info['id']
            mask = mask_segmentations[i]  # 使用原始分割数据
            
            if mask_id in selected_ids:
                color = [255, 255, 255]  # 白色表示选中
            else:
                color = colors[i % len(colors)]
            
            # 在overlay上绘制mask
            mask_area = mask > 0
            overlay[mask_area] = overlay[mask_area] * 0.6 + np.array(color) * 0.4
        
        # 转换回uint8
        return overlay.astype(np.uint8)
        
    except Exception as e:
        print(f"生成mask叠加图失败: {e}")
        return image_np

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """处理图片上传 - 接收前端处理后的base64图片"""
    global predictor
    
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({'error': '没有图片数据'}), 400
        
        session_id = get_session_id()
        init_session(session_id)
        
        # 解析base64图片数据
        image_data = data['image_data']
        width = data['width']
        height = data['height']
        
        # 移除data:image/jpeg;base64,前缀
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # 解码base64数据
        image_bytes = base64.b64decode(image_data)
        
        # 确保 uploads 目录存在
        upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # 保存图片文件
        filename = str(uuid.uuid4()) + '.jpg'
        file_path = os.path.join(upload_dir, filename)
        
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        
        print(f"保存处理后图片到: {file_path}")
        print(f"图片尺寸: {width}x{height}")
        
        # 加载图像
        image = Image.open(file_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 转换为numpy数组
        image_np = np.array(image)
        
        # 存储到会话
        sessions[session_id]['image'] = image_np
        sessions[session_id]['image_path'] = file_path
        sessions[session_id]['points'] = []  # 清空之前的点
        sessions[session_id]['masks'] = []   # 清空之前的mask
        sessions[session_id]['selected_masks'] = set()
        sessions[session_id]['mask_segmentations'] = []
        
        # 设置图像到预测器，准备接收点击
        if predictor:
            predictor.set_image(image_np)
        
        # 返回原图的base64用于前端显示和点击
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'original_image': f'data:image/jpeg;base64,{img_b64}',
            'message': '图片上传成功，请点击选择感兴趣的区域（最多5个点）'
        })
        
    except Exception as e:
        return jsonify({'error': f'图片处理失败: {str(e)}'}), 500

@app.route('/merge_masks', methods=['POST'])
def merge_masks():
    """合并选中的mask"""
    try:
        session_id = get_session_id()
        if session_id not in sessions:
            return jsonify({'error': '会话不存在'}), 400
            
        data = request.get_json()
        selected_ids = data.get('mask_ids', [])
        
        if not selected_ids:
            return jsonify({'error': '没有选择mask'}), 400
        
        masks = sessions[session_id]['masks']
        
        # 合并选中的mask轮廓
        merged_contours = []
        total_area = 0
        
        for mask_id in selected_ids:
            mask = next((m for m in masks if m['id'] == mask_id), None)
            if mask:
                merged_contours.extend(mask['contours'])
                total_area += mask['area']
        
        return jsonify({
            'success': True,
            'merged_mask': {
                'contours': merged_contours,
                'area': total_area,
                'count': len(selected_ids)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'合并失败: {str(e)}'}), 500

@app.route('/click_mask', methods=['POST'])
def click_mask():
    """处理mask点击事件"""
    try:
        session_id = get_session_id()
        if session_id not in sessions:
            return jsonify({'error': '会话不存在'}), 400
            
        data = request.get_json()
        x = int(data['x'])
        y = int(data['y'])
        
        masks = sessions[session_id]['masks']
        mask_segmentations = sessions[session_id]['mask_segmentations']
        selected_masks = sessions[session_id]['selected_masks']
        image_np = sessions[session_id]['image']
        
        # 检查点击位置对应哪个mask
        clicked_mask_id = None
        for i, mask in enumerate(mask_segmentations):
            if y < mask.shape[0] and x < mask.shape[1] and mask[y, x]:
                clicked_mask_id = i
                break
        
        if clicked_mask_id is not None:
            # 切换选中状态
            if clicked_mask_id in selected_masks:
                selected_masks.remove(clicked_mask_id)
            else:
                selected_masks.add(clicked_mask_id)
            
            print(f"点击了mask {clicked_mask_id}, 当前选中: {selected_masks}")
            
            # 重新生成渲染图
            overlay_image = generate_mask_overlay(image_np, masks, mask_segmentations, selected_masks)
            
            # 转换为base64
            overlay_pil = Image.fromarray(overlay_image)
            buffered = BytesIO()
            overlay_pil.save(buffered, format="JPEG")
            overlay_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'overlay_image': f'data:image/jpeg;base64,{overlay_b64}',
                'clicked_mask': clicked_mask_id,
                'selected_count': len(selected_masks)
            })
        else:
            return jsonify({
                'success': False,
                'message': '没有点击到任何区域'
            })
            
    except Exception as e:
        return jsonify({'error': f'点击处理失败: {str(e)}'}), 500

@app.route('/segment_with_points', methods=['POST'])
def segment_with_points():
    """基于用户选择的点进行分割"""
    global predictor
    
    try:
        session_id = get_session_id()
        if session_id not in sessions:
            return jsonify({'error': '会话不存在'}), 400
            
        data = request.get_json()
        points = data.get('points', [])
        
        if not points:
            return jsonify({'error': '没有选择点'}), 400
            
        if len(points) > 5:
            return jsonify({'error': '最多只能选择5个点'}), 400
        
        image_np = sessions[session_id]['image']
        
        print(f"开始基于 {len(points)} 个点进行分割...")
        
        # 对每个点生成mask
        all_masks = []
        for i, point in enumerate(points):
            x, y = int(point['x']), int(point['y'])
            print(f"处理点 {i+1}: ({x}, {y})")
            
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[x, y]]),
                point_labels=np.array([1]),
                multimask_output=False
            )
            
            if len(masks) > 0:
                all_masks.append(masks[0])
        
        if not all_masks:
            return jsonify({'error': '未能生成任何分割结果'}), 400
        
        # 合并所有mask
        combined_mask = np.zeros_like(image_np[:,:,0], dtype=bool)
        for mask in all_masks:
            # 使用 logical_or 确保类型安全
            combined_mask = np.logical_or(combined_mask, mask)
        
        print(f"合并后的mask覆盖像素数: {np.sum(combined_mask)}")
        
        # 生成白色叠加渲染图
        overlay = image_np.copy().astype(np.float32)
        white_color = np.array([255, 255, 255])
        
        # 将mask区域变为白色（半透明叠加）
        overlay[combined_mask] = overlay[combined_mask] * 0.5 + white_color * 0.5
        
        # 转换回uint8
        overlay = overlay.astype(np.uint8)
        
        # 转换为base64
        overlay_pil = Image.fromarray(overlay)
        buffered = BytesIO()
        overlay_pil.save(buffered, format="JPEG")
        overlay_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        # 存储结果
        sessions[session_id]['final_mask'] = combined_mask
        
        return jsonify({
            'success': True,
            'result_image': f'data:image/jpeg;base64,{overlay_b64}',
            'points_count': len(points),
            'mask_area': int(np.sum(combined_mask)),
            'message': f'基于{len(points)}个点的分割完成'
        })
        
    except Exception as e:
        print(f"分割错误: {e}")
        return jsonify({'error': f'分割失败: {str(e)}'}), 500

@app.route('/add_point', methods=['POST'])
def add_point():
    """添加新的点击点"""
    global predictor
    
    try:
        session_id = get_session_id()
        if session_id not in sessions or sessions[session_id]['image'] is None:
            return jsonify({'error': '未上传图片'}), 400
            
        data = request.get_json()
        x = int(data['x'])
        y = int(data['y'])
        label = int(data.get('label', 1))  # 1=正向点, 0=负向点
        
        # 添加点到会话
        point_id = str(uuid.uuid4())
        sessions[session_id]['points'].append({
            'id': point_id,
            'x': x,
            'y': y,
            'label': label
        })
        
        return jsonify({
            'success': True,
            'point_id': point_id,
            'total_points': len(sessions[session_id]['points'])
        })
        
    except Exception as e:
        return jsonify({'error': f'添加点失败: {str(e)}'}), 500

@app.route('/remove_point', methods=['POST'])
def remove_point():
    """移除指定点"""
    try:
        session_id = get_session_id()
        if session_id not in sessions:
            return jsonify({'error': '会话不存在'}), 400
            
        data = request.get_json()
        point_id = data.get('point_id')
        
        if point_id:
            # 移除指定点
            sessions[session_id]['points'] = [
                p for p in sessions[session_id]['points'] if p['id'] != point_id
            ]
        else:
            # 移除最后一个点
            if sessions[session_id]['points']:
                sessions[session_id]['points'].pop()
        
        return jsonify({
            'success': True,
            'total_points': len(sessions[session_id]['points'])
        })
        
    except Exception as e:
        return jsonify({'error': f'移除点失败: {str(e)}'}), 500

@app.route('/clear_points', methods=['POST'])
def clear_points():
    """清除所有点"""
    try:
        session_id = get_session_id()
        if session_id not in sessions:
            return jsonify({'error': '会话不存在'}), 400
            
        sessions[session_id]['points'] = []
        sessions[session_id]['masks'] = []
        sessions[session_id]['selected_masks'] = []
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': f'清除点失败: {str(e)}'}), 500

@app.route('/predict_multi', methods=['POST'])
def predict_multi_masks():
    """基于所有点预测多个候选mask"""
    global predictor
    
    try:
        session_id = get_session_id()
        if session_id not in sessions or sessions[session_id]['image'] is None:
            return jsonify({'error': '未上传图片'}), 400
            
        points = sessions[session_id]['points']
        if not points:
            return jsonify({'error': '没有点击点'}), 400
        
        # 准备输入数据
        input_points = np.array([[p['x'], p['y']] for p in points])
        input_labels = np.array([p['label'] for p in points])
        
        # 预测多个mask（固定返回3个）
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # 只返回得分最高的mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        contours = extract_contours(best_mask)
        mask_data = {
            'contours': contours,
            'score': float(best_score)
        }
        
        sessions[session_id]['masks'] = [mask_data] if contours else []
        sessions[session_id]['selected_masks'] = [0] if contours else []
        
        return jsonify({
            'success': True,
            'mask': mask_data if contours else None,
            'points': points
        })
        
    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500


@app.route('/predict_hover', methods=['POST'])
def predict_hover():
    """专门用于悬浮预览的预测，不修改会话状态"""
    global predictor
    
    try:
        session_id = get_session_id()
        if session_id not in sessions or sessions[session_id]['image'] is None:
            return jsonify({'error': '未上传图片'}), 400
            
        data = request.get_json()
        x = int(data['x'])
        y = int(data['y'])
        
        # 使用SAM 2进行简单预测，不影响会话状态
        input_point = np.array([[x, y]])
        input_label = np.array([1])  # 1表示前景点
        
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        
        # 选择最佳掩码
        best_mask = masks[0]
        
        # 提取轮廓
        contours = extract_contours(best_mask)
        
        return jsonify({
            'success': True,
            'contours': contours,
            'score': float(scores[0])
        })
        
    except Exception as e:
        return jsonify({'error': f'悬浮预测失败: {str(e)}'}), 500

@app.route('/get_session_state', methods=['GET'])
def get_session_state():
    """获取当前会话状态"""
    try:
        session_id = get_session_id()
        if session_id not in sessions:
            return jsonify({'error': '会话不存在'}), 400
            
        session_data = sessions[session_id]
        
        return jsonify({
            'success': True,
            'points': session_data['points'],
            'masks': session_data['masks'],
            'selected_masks': session_data['selected_masks'],
            'has_image': session_data['image'] is not None
        })
        
    except Exception as e:
        return jsonify({'error': f'获取会话状态失败: {str(e)}'}), 500

@app.route('/export_masks', methods=['POST'])
def export_masks():
    """导出选中的masks为PNG"""
    try:
        session_id = get_session_id()
        if session_id not in sessions or sessions[session_id]['image'] is None:
            return jsonify({'error': '未上传图片'}), 400
            
        image = sessions[session_id]['image']
        masks_data = sessions[session_id]['masks']
        
        if not masks_data:
            return jsonify({'error': '没有可用的mask'}), 400
        
        # 获取唯一的mask轮廓数据
        mask_contours = masks_data[0]['contours']
        
        return jsonify({
            'success': True,
            'contours': mask_contours,
            'image_size': {'width': image.shape[1], 'height': image.shape[0]}
        })
        
    except Exception as e:
        return jsonify({'error': f'导出失败: {str(e)}'}), 500

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """切换模型"""
    global predictor
    
    try:
        data = request.get_json()
        model_type = data.get('model_type')
        
        if not model_type:
            return jsonify({'error': '模型类型不能为空'}), 400
            
        if model_type not in MODELS_CONFIG:
            return jsonify({'error': f'不支持的模型类型: {model_type}'}), 400
        
        # 释放当前模型（如果存在）
        if predictor is not None:
            del predictor
            predictor = None
            
        # 清空GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 加载新模型
        success = init_sam2_model(model_type)
        
        if success:
            # 清空所有会话数据，因为模型变了
            sessions.clear()
            
            return jsonify({
                'success': True,
                'model_type': model_type,
                'model_name': MODELS_CONFIG[model_type]['name'],
                'message': f'成功切换到 {MODELS_CONFIG[model_type]["name"]}'
            })
        else:
            return jsonify({'error': '模型加载失败，请检查模型文件是否存在'}), 500
            
    except Exception as e:
        return jsonify({'error': f'切换模型失败: {str(e)}'}), 500

@app.route('/get_models')
def get_models():
    """获取所有可用模型"""
    try:
        models = []
        for model_type, config in MODELS_CONFIG.items():
            # 检查模型文件是否存在（配置文件检查是可选的）
            model_exists = os.path.exists(config['checkpoint'])
            config_exists = os.path.exists(config['config']) if 'config' in config else True
            
            models.append({
                'type': model_type,
                'name': config['name'],
                'available': model_exists,  # 只要模型文件存在就算可用
                'current': model_type == current_model,
                'checkpoint_path': config['checkpoint'],
                'config_path': config.get('config', '')
            })
        
        return jsonify({
            'success': True,
            'models': models,
            'current_model': current_model
        })
        
    except Exception as e:
        return jsonify({'error': f'获取模型列表失败: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'model_loaded': predictor is not None,
        'current_model': current_model,
        'model_name': MODELS_CONFIG[current_model]['name'] if current_model in MODELS_CONFIG else 'Unknown',
        'device': str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    })

if __name__ == '__main__':
    print("正在初始化SAM 2模型...")
    if init_sam2_model():
        print("启动Flask应用...")
        app.run(debug=True, host='0.0.0.0', port=3000)
    else:
        print("模型初始化失败，请检查模型文件")