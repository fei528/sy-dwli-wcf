import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def gaussian(x, sigma=100.0):
    """高斯函数: exp(-x²/(2σ²))"""
    return np.exp(-x**2 / (2 * sigma**2))

def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )
    return o

def enhanced_iou_batch(bboxes1, bboxes2, sigma_factor=1.0):
    """
    增强版IoU：IoU × 边距离高斯影响因子（自适应sigma）
    
    Args:
        bboxes1, bboxes2: 边界框数组
        sigma_factor: sigma调节因子，默认1.0
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # 计算传统IoU
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    iou = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )
    
    # 计算每个边界框的边长
    w1 = bboxes1[..., 2] - bboxes1[..., 0]  # bboxes1的宽度
    h1 = bboxes1[..., 3] - bboxes1[..., 1]  # bboxes1的高度
    w2 = bboxes2[..., 2] - bboxes2[..., 0]  # bboxes2的宽度
    h2 = bboxes2[..., 3] - bboxes2[..., 1]  # bboxes2的高度
    
    # 找到每对框中最短的边作为自适应sigma
    min_edge1 = np.minimum(w1, h1)  # bboxes1中每个框的最短边
    min_edge2 = np.minimum(w2, h2)  # bboxes2中每个框的最短边
    adaptive_sigma = np.minimum(min_edge1, min_edge2) * sigma_factor  # 取两者中更小的
    
    # 计算四条边的距离
    left_dist = np.abs(bboxes1[..., 0] - bboxes2[..., 0])      # 左边距离
    top_dist = np.abs(bboxes1[..., 1] - bboxes2[..., 1])       # 上边距离
    right_dist = np.abs(bboxes1[..., 2] - bboxes2[..., 2])     # 右边距离
    bottom_dist = np.abs(bboxes1[..., 3] - bboxes2[..., 3])    # 下边距离
    
    # 计算高斯得分（使用自适应sigma）
    left_score = gaussian(left_dist, adaptive_sigma)
    top_score = gaussian(top_dist, adaptive_sigma)
    right_score = gaussian(right_dist, adaptive_sigma)
    bottom_score = gaussian(bottom_dist, adaptive_sigma)
    
    # 影响因子：四个得分的平均值
    influence_factor = (left_score + top_score + right_score + bottom_score) / 4.0
    
    # 增强IoU
    enhanced_iou = influence_factor * iou
    
    return enhanced_iou, iou, influence_factor, adaptive_sigma

def visualize_iou_results(bboxes1, bboxes2, sigma_factor=1.0):
    """
    简洁的IoU计算结果可视化（自适应sigma版本）
    """
    enhanced_iou, original_iou, influence_factor, adaptive_sigma = enhanced_iou_batch(bboxes1, bboxes2, sigma_factor)
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 绘制边界框
    ax1 = axes[0, 0]
    draw_bboxes_with_labels(ax1, bboxes1, bboxes2)
    ax1.set_title('Bounding Boxes', fontsize=16, fontweight='bold')
    
    # 2. 原始IoU热力图
    ax2 = axes[0, 1]
    im1 = ax2.imshow(original_iou, cmap='Blues', vmin=0, vmax=1)
    add_values_to_heatmap(ax2, original_iou, fontsize=14)
    plt.colorbar(im1, ax=ax2, shrink=0.8)
    ax2.set_title('Original IoU', fontsize=16, fontweight='bold')
    ax2.set_xlabel('bboxes2 index')
    ax2.set_ylabel('bboxes1 index')
    
    # 3. 增强IoU热力图
    ax3 = axes[0, 2]
    im2 = ax3.imshow(enhanced_iou, cmap='Reds', vmin=0, vmax=1)
    add_values_to_heatmap(ax3, enhanced_iou, fontsize=14)
    plt.colorbar(im2, ax=ax3, shrink=0.8)
    ax3.set_title('Enhanced IoU', fontsize=16, fontweight='bold')
    ax3.set_xlabel('bboxes2 index')
    ax3.set_ylabel('bboxes1 index')
    
    # 4. 影响因子热力图
    ax4 = axes[1, 0]
    im3 = ax4.imshow(influence_factor, cmap='Greens', vmin=0, vmax=1)
    add_values_to_heatmap(ax4, influence_factor, fontsize=14)
    plt.colorbar(im3, ax=ax4, shrink=0.8)
    ax4.set_title('Influence Factor', fontsize=16, fontweight='bold')
    ax4.set_xlabel('bboxes2 index')
    ax4.set_ylabel('bboxes1 index')
    
    # 5. 数值对比表格
    ax5 = axes[1, 1]
    create_comparison_table(ax5, original_iou, enhanced_iou, influence_factor)
    ax5.set_title('Numerical Comparison', fontsize=16, fontweight='bold')
    
    # 6. IoU差异图
    ax6 = axes[1, 2]
    diff_matrix = enhanced_iou - original_iou
    im4 = ax6.imshow(diff_matrix, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    add_values_to_heatmap(ax6, diff_matrix, fontsize=14, format_str='{:.3f}')
    plt.colorbar(im4, ax=ax6, shrink=0.8)
    ax6.set_title('Difference (Enhanced - Original)', fontsize=16, fontweight='bold')
    ax6.set_xlabel('bboxes2 index')
    ax6.set_ylabel('bboxes1 index')
    
    # 调整布局
    plt.tight_layout(pad=3.0)
    
    # 保存图片
    plt.savefig('iou_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("IoU results visualization saved as 'iou_results.png'")
    
    # 打印自适应sigma信息
    print("\nAdaptive Sigma Matrix:")
    print(adaptive_sigma)

def draw_bboxes_with_labels(ax, bboxes1, bboxes2):
    """绘制带标签的边界框"""
    # 计算合适的坐标范围
    all_coords = np.concatenate([bboxes1, bboxes2], axis=0)
    x_min, y_min = all_coords[:, [0, 1]].min(axis=0) - 20
    x_max, y_max = all_coords[:, [2, 3]].max(axis=0) + 20
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # 翻转y轴
    
    # 绘制bboxes1 (蓝色)
    for i, bbox in enumerate(bboxes1):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor='blue', facecolor='lightblue', alpha=0.4)
        ax.add_patch(rect)
        # 中心位置添加标签
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(cx, cy, f'B1_{i}', fontsize=14, color='blue', weight='bold', 
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 绘制bboxes2 (红色)
    for i, bbox in enumerate(bboxes2):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor='red', facecolor='lightcoral', alpha=0.4)
        ax.add_patch(rect)
        # 中心位置添加标签
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(cx, cy, f'B2_{i}', fontsize=14, color='red', weight='bold',
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True, alpha=0.3)

def add_values_to_heatmap(ax, matrix, fontsize=12, format_str='{:.3f}'):
    """在热力图上添加数值"""
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = 'white' if value > 0.5 else 'black'
            text = ax.text(j, i, format_str.format(value), 
                         ha="center", va="center", color=color, 
                         fontsize=fontsize, fontweight='bold')

def create_comparison_table(ax, original_iou, enhanced_iou, influence_factor):
    """创建数值对比表格"""
    ax.axis('off')
    
    # 准备表格数据
    rows, cols = original_iou.shape
    table_data = []
    
    # 表头
    headers = ['Pair', 'Original IoU', 'Enhanced IoU', 'Influence Factor', 'Difference']
    
    # 填充数据
    for i in range(rows):
        for j in range(cols):
            pair_name = f'B1_{i} - B2_{j}'
            orig = original_iou[i, j]
            enh = enhanced_iou[i, j]
            inf = influence_factor[i, j]
            diff = enh - orig
            
            table_data.append([
                pair_name,
                f'{orig:.3f}',
                f'{enh:.3f}',
                f'{inf:.3f}',
                f'{diff:+.3f}'
            ])
    
    # 创建表格
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置数据行颜色
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')

def test_enhanced_iou():
    """测试函数 - 分为小、中、大三个案例"""
    
    print("=" * 80)
    print("Enhanced IoU Test Results with Adaptive Sigma (Separate Test Cases)")
    print("=" * 80)
    
    # 测试案例1：小尺寸框
    print("\n1. Small Bboxes Test Case:")
    print("-" * 40)
    test_small_cases()
    
    # 测试案例2：中等尺寸框
    print("\n2. Medium Bboxes Test Case:")
    print("-" * 40)
    test_medium_cases()
    
    # 测试案例3：大尺寸框
    print("\n3. Large Bboxes Test Case:")
    print("-" * 40)
    test_large_cases()
    
    print("=" * 80)
    print("All test cases completed! Generated 3 visualization files:")
    print("- small_boxes_iou.png")
    print("- medium_boxes_iou.png") 
    print("- large_boxes_iou.png")

def test_small_cases():
    """小尺寸框测试案例"""
    # 小框：20-60像素尺寸
    bboxes1_small = np.array([
        [100, 100, 140, 120],   # 40x20的小框 (最短边=20)
        [200, 200, 240, 240],   # 40x40的小框 (最短边=40)
        [300, 300, 360, 330],   # 60x30的小框 (最短边=30)
    ])
    
    bboxes2_small = np.array([
        [100, 100, 140, 120],   # 完全重合
        [105, 105, 145, 125],   # 5像素偏移
        [115, 115, 155, 135],   # 15像素偏移
        [130, 130, 170, 150],   # 30像素偏移
    ])
    
    enhanced_iou, original_iou, influence_factor, adaptive_sigma = enhanced_iou_batch(bboxes1_small, bboxes2_small, sigma_factor=1.0)
    
    print("Small Boxes Dimensions:")
    for i, bbox in enumerate(bboxes1_small):
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        print(f"  B1_{i}: {w}x{h} (shortest edge = {min(w,h)})")
    
    print("Adaptive Sigma Matrix:")
    print(adaptive_sigma)
    print("Original IoU Matrix:")
    print(original_iou)
    print("Enhanced IoU Matrix:")
    print(enhanced_iou)
    
    # 生成可视化
    visualize_iou_results_with_title(bboxes1_small, bboxes2_small, 
                                   "Small Bboxes (20-60px)", "small_boxes_iou.png", sigma_factor=1.0)

def test_medium_cases():
    """中等尺寸框测试案例"""
    # 中框：80-150像素尺寸
    bboxes1_medium = np.array([
        [100, 100, 200, 180],   # 100x80的框 (最短边=80)
        [300, 300, 450, 420],   # 150x120的框 (最短边=120)
        [500, 500, 600, 650],   # 100x150的框 (最短边=100)
    ])
    
    bboxes2_medium = np.array([
        [100, 100, 200, 180],   # 完全重合
        [120, 120, 220, 200],   # 20像素偏移
        [150, 150, 250, 230],   # 50像素偏移
        [200, 200, 300, 280],   # 100像素偏移
        [280, 280, 380, 360],   # 180像素偏移
    ])
    
    enhanced_iou, original_iou, influence_factor, adaptive_sigma = enhanced_iou_batch(bboxes1_medium, bboxes2_medium, sigma_factor=1.0)
    
    print("Medium Boxes Dimensions:")
    for i, bbox in enumerate(bboxes1_medium):
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        print(f"  B1_{i}: {w}x{h} (shortest edge = {min(w,h)})")
    
    print("Adaptive Sigma Matrix:")
    print(adaptive_sigma)
    print("Original IoU Matrix:")
    print(original_iou)
    print("Enhanced IoU Matrix:")
    print(enhanced_iou)
    
    # 生成可视化
    visualize_iou_results_with_title(bboxes1_medium, bboxes2_medium, 
                                   "Medium Bboxes (80-150px)", "medium_boxes_iou.png", sigma_factor=1.0)

def test_large_cases():
    """大尺寸框测试案例"""
    # 大框：200-400像素尺寸
    bboxes1_large = np.array([
        [100, 100, 400, 350],   # 300x250的大框 (最短边=250)
        [500, 500, 900, 700],   # 400x200的大框 (最短边=200)
        [200, 800, 600, 1200],  # 400x400的大框 (最短边=400)
    ])
    
    bboxes2_large = np.array([
        [100, 100, 400, 350],   # 完全重合
        [150, 150, 450, 400],   # 50像素偏移
        [250, 250, 550, 500],   # 150像素偏移
        [400, 400, 700, 650],   # 300像素偏移
        [600, 600, 900, 850],   # 500像素偏移
    ])
    
    enhanced_iou, original_iou, influence_factor, adaptive_sigma = enhanced_iou_batch(bboxes1_large, bboxes2_large, sigma_factor=1.0)
    
    print("Large Boxes Dimensions:")
    for i, bbox in enumerate(bboxes1_large):
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        print(f"  B1_{i}: {w}x{h} (shortest edge = {min(w,h)})")
    
    print("Adaptive Sigma Matrix:")
    print(adaptive_sigma)
    print("Original IoU Matrix:")
    print(original_iou)
    print("Enhanced IoU Matrix:")
    print(enhanced_iou)
    
    # 生成可视化
    visualize_iou_results_with_title(bboxes1_large, bboxes2_large, 
                                   "Large Bboxes (200-400px)", "large_boxes_iou.png", sigma_factor=1.0)

def visualize_iou_results_with_title(bboxes1, bboxes2, title, filename, sigma_factor=1.0):
    """
    生成带标题的IoU可视化结果
    """
    enhanced_iou, original_iou, influence_factor, adaptive_sigma = enhanced_iou_batch(bboxes1, bboxes2, sigma_factor)
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Enhanced IoU Analysis: {title}', fontsize=20, fontweight='bold')
    
    # 1. 绘制边界框
    ax1 = axes[0, 0]
    draw_bboxes_with_labels(ax1, bboxes1, bboxes2)
    ax1.set_title('Bounding Boxes', fontsize=16, fontweight='bold')
    
    # 2. 原始IoU热力图
    ax2 = axes[0, 1]
    im1 = ax2.imshow(original_iou, cmap='Blues', vmin=0, vmax=1)
    add_values_to_heatmap(ax2, original_iou, fontsize=12)
    plt.colorbar(im1, ax=ax2, shrink=0.8)
    ax2.set_title('Original IoU', fontsize=16, fontweight='bold')
    ax2.set_xlabel('bboxes2 index')
    ax2.set_ylabel('bboxes1 index')
    
    # 3. 增强IoU热力图
    ax3 = axes[0, 2]
    im2 = ax3.imshow(enhanced_iou, cmap='Reds', vmin=0, vmax=1)
    add_values_to_heatmap(ax3, enhanced_iou, fontsize=12)
    plt.colorbar(im2, ax=ax3, shrink=0.8)
    ax3.set_title('Enhanced IoU', fontsize=16, fontweight='bold')
    ax3.set_xlabel('bboxes2 index')
    ax3.set_ylabel('bboxes1 index')
    
    # 4. 影响因子热力图
    ax4 = axes[1, 0]
    im3 = ax4.imshow(influence_factor, cmap='Greens', vmin=0, vmax=1)
    add_values_to_heatmap(ax4, influence_factor, fontsize=12)
    plt.colorbar(im3, ax=ax4, shrink=0.8)
    ax4.set_title('Influence Factor', fontsize=16, fontweight='bold')
    ax4.set_xlabel('bboxes2 index')
    ax4.set_ylabel('bboxes1 index')
    
    # 5. 自适应Sigma热力图
    ax5 = axes[1, 1]
    im4 = ax5.imshow(adaptive_sigma, cmap='Purples', vmin=0, vmax=adaptive_sigma.max())
    add_values_to_heatmap(ax5, adaptive_sigma, fontsize=12, format_str='{:.0f}')
    plt.colorbar(im4, ax=ax5, shrink=0.8)
    ax5.set_title('Adaptive Sigma', fontsize=16, fontweight='bold')
    ax5.set_xlabel('bboxes2 index')
    ax5.set_ylabel('bboxes1 index')
    
    # 6. 数值对比表格
    ax6 = axes[1, 2]
    create_comparison_table_adaptive(ax6, original_iou, enhanced_iou, influence_factor, adaptive_sigma)
    ax6.set_title('Detailed Comparison', fontsize=16, fontweight='bold')
    
    # 调整布局
    plt.tight_layout(pad=3.0)
    
    # 保存图片
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved as '{filename}'")

def create_comparison_table_adaptive(ax, original_iou, enhanced_iou, influence_factor, adaptive_sigma):
    """创建包含自适应sigma的对比表格"""
    ax.axis('off')
    
    # 准备表格数据
    rows, cols = original_iou.shape
    table_data = []
    
    # 表头
    headers = ['Pair', 'Sigma', 'Orig IoU', 'Enh IoU', 'Influence', 'Diff']
    
    # 填充数据
    for i in range(rows):
        for j in range(cols):
            pair_name = f'B1_{i}-B2_{j}'
            sigma = adaptive_sigma[i, j]
            orig = original_iou[i, j]
            enh = enhanced_iou[i, j]
            inf = influence_factor[i, j]
            diff = enh - orig
            
            table_data.append([
                pair_name,
                f'{sigma:.0f}',
                f'{orig:.3f}',
                f'{enh:.3f}',
                f'{inf:.3f}',
                f'{diff:+.3f}'
            ])
    
    # 创建表格
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置数据行颜色
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')

if __name__ == "__main__":
    # 设置matplotlib后端
    import matplotlib
    matplotlib.use('Agg')
    
    print("Starting Enhanced IoU Test...")
    test_enhanced_iou()
    print("Visualization complete!")