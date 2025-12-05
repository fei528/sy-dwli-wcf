import cv2
import numpy as np

def pyramid_ecc(src, dst, warp_mode=cv2.MOTION_EUCLIDEAN, eps=1e-5,
                max_iter=100, pyramid_levels=4, align=False, 
                pyramid_type='gaussian', preprocess=True):
    """
    基于图像金字塔的ECC算法，显著提升性能和鲁棒性
    
    Parameters:
    -----------
    pyramid_levels : int
        金字塔层数，建议3-5层
    pyramid_type : str
        'gaussian' - 高斯金字塔（推荐）
        'laplacian' - 拉普拉斯金字塔
    """
    assert src.shape == dst.shape, "源图像和目标图像必须具有相同的格式!"

    # 转换为灰度图
    if src.ndim == 3:
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    else:
        src_gray = src.copy()
        dst_gray = dst.copy()

    # 图像预处理
    if preprocess:
        src_gray = cv2.GaussianBlur(src_gray, (3, 3), 0)
        dst_gray = cv2.GaussianBlur(dst_gray, (3, 3), 0)
        
        # 自适应直方图均衡化，比普通均衡化效果更好
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        src_gray = clahe.apply(src_gray)
        dst_gray = clahe.apply(dst_gray)

    # 构建图像金字塔
    if pyramid_type == 'gaussian':
        src_pyramid = build_gaussian_pyramid(src_gray, pyramid_levels)
        dst_pyramid = build_gaussian_pyramid(dst_gray, pyramid_levels)
    else:
        src_pyramid = build_laplacian_pyramid(src_gray, pyramid_levels)
        dst_pyramid = build_laplacian_pyramid(dst_gray, pyramid_levels)

    # 初始化变换矩阵
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # 从最粗糙层开始，逐层精化
    total_cc = 0
    for level in range(pyramid_levels-1, -1, -1):
        # 当前层的图像
        src_level = src_pyramid[level]
        dst_level = dst_pyramid[level]
        
        # 调整迭代次数：粗糙层少迭代，精细层多迭代
        level_max_iter = max(10, max_iter // (level + 1))
        
        # 调整精度要求：粗糙层要求低，精细层要求高
        level_eps = eps * (2 ** level)
        
        # 定义当前层的终止条件
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                   level_max_iter, level_eps)

        print(f"金字塔第{level}层: 尺寸{src_level.shape}, 迭代{level_max_iter}次, eps={level_eps:.6f}")

        try:
            # 在当前层运行ECC
            (cc, warp_matrix) = cv2.findTransformECC(
                src_level, dst_level, warp_matrix, warp_mode, criteria)
            
            total_cc += cc
            print(f"  第{level}层收敛，相关系数: {cc:.4f}")
            
            # 如果不是最后一层，需要将变换矩阵放缩到下一层
            if level > 0:
                scale_factor = 2.0  # 下一层是当前层的2倍大小
                warp_matrix = scale_warp_matrix(warp_matrix, scale_factor, warp_mode)
                
        except cv2.error as e:
            print(f"第{level}层ECC失败: {e}")
            # 如果当前层失败，继续下一层，但可能精度会下降
            if level > 0:
                scale_factor = 2.0
                warp_matrix = scale_warp_matrix(warp_matrix, scale_factor, warp_mode)
            continue

    avg_cc = total_cc / pyramid_levels
    print(f"金字塔ECC完成，平均相关系数: {avg_cc:.4f}")

    if align:
        sz = src_gray.shape
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            src_aligned = cv2.warpPerspective(src_gray, warp_matrix, 
                                            (sz[1], sz[0]), flags=cv2.INTER_LINEAR)
        else:
            src_aligned = cv2.warpAffine(src_gray, warp_matrix, 
                                       (sz[1], sz[0]), flags=cv2.INTER_LINEAR)
        return warp_matrix, src_aligned
    else:
        return warp_matrix, None


def build_gaussian_pyramid(image, levels):
    """构建高斯金字塔"""
    pyramid = [image]
    current = image
    
    for i in range(levels - 1):
        # 高斯模糊后下采样
        current = cv2.pyrDown(current)
        pyramid.append(current)
    
    return pyramid


def build_laplacian_pyramid(image, levels):
    """构建拉普拉斯金字塔 - 保留更多细节信息"""
    gaussian_pyramid = build_gaussian_pyramid(image, levels)
    laplacian_pyramid = []
    
    for i in range(levels - 1):
        # 计算拉普拉斯层
        expanded = cv2.pyrUp(gaussian_pyramid[i + 1])
        # 确保尺寸匹配
        if expanded.shape != gaussian_pyramid[i].shape:
            expanded = cv2.resize(expanded, (gaussian_pyramid[i].shape[1], 
                                           gaussian_pyramid[i].shape[0]))
        laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
        laplacian_pyramid.append(laplacian)
    
    # 最后一层就是高斯金字塔的顶层
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    return laplacian_pyramid


def scale_warp_matrix(warp_matrix, scale_factor, warp_mode):
    """将变换矩阵从一个尺度缩放到另一个尺度"""
    scaled_matrix = warp_matrix.copy()
    
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # 对于单应性矩阵，需要特殊处理
        # H_new = S * H * S^(-1), 其中S是缩放矩阵
        S = np.array([[scale_factor, 0, 0],
                      [0, scale_factor, 0],
                      [0, 0, 1]], dtype=np.float32)
        S_inv = np.array([[1/scale_factor, 0, 0],
                          [0, 1/scale_factor, 0],
                          [0, 0, 1]], dtype=np.float32)
        scaled_matrix = S @ warp_matrix @ S_inv
    else:
        # 对于仿射变换，只需要缩放平移分量
        scaled_matrix[0, 2] *= scale_factor
        scaled_matrix[1, 2] *= scale_factor
    
    return scaled_matrix


def adaptive_pyramid_ecc(src, dst, motion_magnitude='auto', **kwargs):
    """
    自适应金字塔ECC - 根据运动幅度自动调整金字塔参数
    """
    # 快速估计运动幅度
    if motion_magnitude == 'auto':
        motion_mag = estimate_motion_magnitude(src, dst)
        print(f"估计运动幅度: {motion_mag:.2f}")
    else:
        motion_mag = motion_magnitude
    
    # 根据运动幅度自适应调整参数
    if motion_mag > 50:  # 大幅运动
        pyramid_levels = 5
        max_iter = 150
        eps = 1e-6
        print("检测到大幅运动，使用5层金字塔")
    elif motion_mag > 20:  # 中等运动
        pyramid_levels = 4
        max_iter = 100
        eps = 1e-5
        print("检测到中等运动，使用4层金字塔")
    else:  # 小幅运动
        pyramid_levels = 3
        max_iter = 50
        eps = 1e-4
        print("检测到小幅运动，使用3层金字塔")
    
    return pyramid_ecc(src, dst, pyramid_levels=pyramid_levels, 
                      max_iter=max_iter, eps=eps, **kwargs)


def estimate_motion_magnitude(src, dst):
    """快速估计两帧之间的运动幅度"""
    # 转换为灰度图
    if src.ndim == 3:
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    else:
        src_gray = src
        dst_gray = dst
    
    # 使用稀疏光流快速估计
    corners = cv2.goodFeaturesToTrack(src_gray, maxCorners=100, 
                                    qualityLevel=0.01, minDistance=10)
    
    if corners is not None:
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        next_corners, status, error = cv2.calcOpticalFlowPyrLK(
            src_gray, dst_gray, corners, None, **lk_params)
        
        # 计算平均位移
        good_corners = corners[status == 1]
        good_next = next_corners[status == 1]
        
        if len(good_corners) > 0:
            displacements = np.linalg.norm(good_next - good_corners, axis=1)
            return np.median(displacements)  # 使用中位数更鲁棒
    
    return 10  # 默认中等运动


def compare_ecc_methods(src, dst):
    """比较不同ECC方法的性能"""
    import time
    
    methods = {
        '原始ECC': lambda: ecc(src, dst, scale=0.3, align=True),
        '单层ECC': lambda: pyramid_ecc(src, dst, pyramid_levels=1, align=True),
        '3层金字塔': lambda: pyramid_ecc(src, dst, pyramid_levels=3, align=True),
        '4层金字塔': lambda: pyramid_ecc(src, dst, pyramid_levels=4, align=True),
        '自适应金字塔': lambda: adaptive_pyramid_ecc(src, dst, align=True)
    }
    
    results = {}
    for name, method in methods.items():
        try:
            start = time.time()
            warp_matrix, aligned = method()
            elapsed = time.time() - start
            
            # 计算对齐质量（使用SSIM）
            if aligned is not None:
                dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY) if dst.ndim == 3 else dst
                ssim_score = cv2.matchTemplate(aligned, dst_gray, cv2.TM_CCOEFF_NORMED)[0, 0]
            else:
                ssim_score = 0
            
            results[name] = {
                'time': elapsed,
                'quality': ssim_score,
                'warp_matrix': warp_matrix
            }
            
            print(f"{name}: 耗时 {elapsed:.4f}s, 质量 {ssim_score:.4f}")
            
        except Exception as e:
            print(f"{name} 失败: {e}")
            results[name] = {'time': float('inf'), 'quality': 0, 'warp_matrix': None}
    
    return results


# 使用示例
if __name__ == "__main__":
    # 假设你有两帧图像
    src = cv2.imread('/workspace/Deep-OC-SORT/data/dancetrack/train/dancetrack0001/img1/00000001.jpg')
    dst = cv2.imread('/workspace/Deep-OC-SORT/data/dancetrack/train/dancetrack0001/img1/00000002.jpg')
    
    # 使用自适应金字塔ECC
    warp_matrix, aligned = adaptive_pyramid_ecc(src, dst, align=True)
    
    # 或者直接使用固定层数的金字塔
    # warp_matrix, aligned = pyramid_ecc(src, dst, pyramid_levels=4, align=True)
    
    # 性能比较
    results = compare_ecc_methods(src, dst)
    pass