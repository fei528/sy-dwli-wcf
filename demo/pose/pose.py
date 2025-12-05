import cv2
import mediapipe as mp
import numpy as np
import os
import time
import warnings

# 抑制所有警告和日志
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

class OptimizedTorsoDetector:
    """优化的躯干检测器 - 平衡速度和精度"""
    
    def __init__(self):
        # 使用最轻量的MediaPipe配置
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,        # 最轻量模型
            smooth_landmarks=False,    # 关闭平滑
            enable_segmentation=False, # 关闭分割
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # 缓存最近的结果
        self.cache = {}
        self.max_cache_size = 100
        
    def preprocess_image(self, image):
        """极速预处理"""
        h, w = image.shape[:2]
        
        # 智能缩放策略
        if max(h, w) > 480:
            # 大图缩放到480px（平衡速度和精度）
            scale = 480 / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 使用最快的插值方法
            small_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            return small_img, scale
        
        return image, 1.0
    
    def detect_torso_bbox(self, image_path, output_path=None):
        """主检测函数"""
        # 检查缓存
        cache_key = f"{image_path}_{os.path.getmtime(image_path)}"
        if cache_key in self.cache:
            bbox = self.cache[cache_key]
            if output_path:
                self.draw_bbox(image_path, bbox, output_path)
            return bbox
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        # 预处理
        processed_img, scale = self.preprocess_image(image)
        
        # 转换为RGB（MediaPipe要求）
        rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        # 姿态检测
        results = self.pose.process(rgb_img)
        
        if not results.pose_landmarks:
            print("未检测到人体姿态")
            return None
        
        # 提取躯干关键点（只用4个核心点）
        landmarks = results.pose_landmarks.landmark
        h, w = processed_img.shape[:2]
        
        # 关键点索引：左肩(11)、右肩(12)、左髋(23)、右髋(24)
        torso_points = []
        for idx in [11, 12, 23, 24]:
            if landmarks[idx].visibility > 0.3:
                x = landmarks[idx].x * w / scale
                y = landmarks[idx].y * h / scale
                torso_points.append((x, y))
        
        if len(torso_points) < 3:
            print("躯干关键点不足")
            return None
        
        # 快速计算边界框
        x_coords = [p[0] for p in torso_points]
        y_coords = [p[1] for p in torso_points]
        
        # 添加最小边距
        padding = 20
        min_x = max(0, int(min(x_coords) - padding))
        min_y = max(0, int(min(y_coords) - padding))
        max_x = min(image.shape[1], int(max(x_coords) + padding))
        max_y = min(image.shape[0], int(max(y_coords) + padding))
        
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
        
        # 缓存结果
        if len(self.cache) >= self.max_cache_size:
            # 清除最旧的缓存
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[cache_key] = bbox
        
        # 绘制结果
        if output_path:
            self.draw_bbox(image_path, bbox, output_path)
        
        return bbox
    
    def draw_bbox(self, image_path, bbox, output_path):
        """绘制边界框"""
        image = cv2.imread(image_path)
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imwrite(output_path, image)


# 全局检测器实例（避免重复初始化）
detector = OptimizedTorsoDetector()


def fast_detect(image_path, output_path=None, show_time=True):
    """
    快速躯干检测
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        show_time: 是否显示处理时间
    
    Returns:
        bbox: [x, y, w, h] 格式的边界框坐标
    """
    start_time = time.time()
    
    # 检测
    bbox = detector.detect_torso_bbox(image_path, output_path)
    
    end_time = time.time()
    cost_ms = (end_time - start_time) * 1000
    
    if bbox:
        print(f"躯干边界框: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
        if show_time:
            print(f"处理时间: {cost_ms:.2f} 毫秒")
        if output_path:
            print(f"保存到: {output_path}")
    
    return bbox


def batch_process(image_dir, output_dir=None):
    """批量处理"""
    if output_dir is None:
        output_dir = image_dir + "_bbox"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像文件
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
    
    print(f"开始批量处理 {len(image_files)} 张图像...")
    
    total_time = 0
    success_count = 0
    
    for i, filename in enumerate(image_files):
        input_path = os.path.join(image_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        start = time.time()
        bbox = fast_detect(input_path, output_path, show_time=False)
        end = time.time()
        
        if bbox:
            success_count += 1
        
        total_time += (end - start)
        
        if (i + 1) % 10 == 0:
            avg_time = (total_time / (i + 1)) * 1000
            print(f"进度: {i+1}/{len(image_files)}, 平均: {avg_time:.1f}ms")
    
    avg_time = (total_time / len(image_files)) * 1000
    print(f"\n批量处理完成:")
    print(f"成功: {success_count}/{len(image_files)}")
    print(f"平均处理时间: {avg_time:.2f} 毫秒")


def benchmark(image_path, iterations=50):
    """性能基准测试"""
    print(f"开始性能测试...")
    
    # 预热
    for _ in range(5):
        detector.detect_torso_bbox(image_path)
    
    times = []
    success_count = 0
    
    for i in range(iterations):
        start = time.time()
        bbox = detector.detect_torso_bbox(image_path)
        end = time.time()
        
        if bbox:
            success_count += 1
            times.append((end - start) * 1000)
    
    if times:
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"\n性能测试结果 ({iterations} 次):")
        print(f"成功率: {success_count}/{iterations} ({success_count/iterations*100:.1f}%)")
        print(f"平均时间: {avg_time:.2f} 毫秒")
        print(f"最快时间: {min_time:.2f} 毫秒")
        print(f"最慢时间: {max_time:.2f} 毫秒")
        
        if avg_time <= 15:
            print("✓ 性能优秀 (≤15ms)")
        elif avg_time <= 30:
            print("○ 性能良好 (≤30ms)")
        else:
            print("△ 需要进一步优化")
    else:
        print("测试失败：无法检测到躯干")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  单张检测: python pose_detection.py <图像路径> [输出路径]")
        print("  批量处理: python pose_detection.py --batch <图像目录> [输出目录]")
        print("  性能测试: python pose_detection.py --benchmark <图像路径>")
        sys.exit(1)
    
    if sys.argv[1] == "--benchmark":
        if len(sys.argv) < 3:
            print("请提供测试图像路径")
            sys.exit(1)
        benchmark(sys.argv[2])
    elif sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("请提供图像目录")
            sys.exit(1)
        input_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        batch_process(input_dir, output_dir)
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        fast_detect(input_path, output_path)