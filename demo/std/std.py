import numpy as np

def read_detection_data(file_path):
    """读取检测框数据文件"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        boxes = []
        for line in lines:
            line = line.strip()
            if line:  # 跳过空行
                # 分割数据，取前两个数值作为宽度和高度
                values = line.split()
                if len(values) >= 2:
                    width = float(values[0])
                    height = float(values[1])
                    boxes.append((width, height))
        
        return boxes
    except Exception as e:
        print(f"读取文件失败: {e}")
        return []

def classify_dataset(boxes):
    """根据宽高比特征分类数据集"""
    if not boxes:
        return "无数据", 0, {}
    
    # 计算宽高比
    ratios = [width / height for width, height in boxes]
    
    # 计算关键统计特征
    ratio_mean = np.mean(ratios)
    ratio_std = np.std(ratios)
    ratio_min = np.min(ratios)
    ratio_max = np.max(ratios)
    ratio_range = ratio_max - ratio_min
    
    features = {
        '样本数量': len(boxes),
        '宽高比均值': round(ratio_mean, 3),
        '宽高比标准差': round(ratio_std, 3),
        '宽高比范围': f"{ratio_min:.3f} ~ {ratio_max:.3f}",
        '范围大小': round(ratio_range, 3)
    }
    
    # 分类逻辑（主要基于标准差）
    if ratio_std < 0.12:
        if ratio_mean < 0.45:
            classification = "数据集1类型"
            confidence = 0.95
            reason = "低标准差 + 低均值：一致性窄长目标"
        else:
            classification = "数据集1类型"
            confidence = 0.75
            reason = "低标准差但均值偏高：一致性目标"
    else:
        if ratio_mean > 0.45:
            classification = "数据集2类型"
            confidence = 0.95
            reason = "高标准差 + 高均值：多样性目标"
        else:
            classification = "数据集2类型"
            confidence = 0.75
            reason = "高标准差：多样性目标"
    
    return classification, confidence, features, reason

def analyze_files(file1_path, file2_path):
    """分析两个文件并输出结果"""
    print("检测框数据集分类分析")
    print("=" * 50)
    
    # 读取文件1
    print(f"\n分析文件1: {file1_path}")
    boxes1 = read_detection_data(file1_path)
    if boxes1:
        class1, conf1, features1, reason1 = classify_dataset(boxes1)
        print(f"分类结果: {class1} (置信度: {conf1:.2f})")
        print(f"判断依据: {reason1}")
        print("数据特征:")
        for key, value in features1.items():
            print(f"  {key}: {value}")
    else:
        print("文件1读取失败或无有效数据")
    
    # 读取文件2
    print(f"\n分析文件2: {file2_path}")
    boxes2 = read_detection_data(file2_path)
    if boxes2:
        class2, conf2, features2, reason2 = classify_dataset(boxes2)
        print(f"分类结果: {class2} (置信度: {conf2:.2f})")
        print(f"判断依据: {reason2}")
        print("数据特征:")
        for key, value in features2.items():
            print(f"  {key}: {value}")
    else:
        print("文件2读取失败或无有效数据")
    
    # 对比分析
    if boxes1 and boxes2:
        print(f"\n对比分析:")
        print("=" * 30)
        std1 = features1['宽高比标准差']
        std2 = features2['宽高比标准差']
        mean1 = features1['宽高比均值']
        mean2 = features2['宽高比均值']
        
        print(f"标准差对比: {std1} vs {std2} (差异: {abs(std2-std1):.3f})")
        print(f"均值对比: {mean1} vs {mean2} (差异: {abs(mean2-mean1):.3f})")
        
        # 明确指出哪个标准差更小
        if std1 < std2:
            print(f"✓ 文件1的标准差更小 ({std1} < {std2})")
            print(f"  → 文件1数据更一致，标准差小了 {((std2-std1)/std1*100):.1f}%")
            print(f"  → 文件2数据更多样，标准差是文件1的 {(std2/std1):.1f} 倍")
        elif std2 < std1:
            print(f"✓ 文件2的标准差更小 ({std2} < {std1})")
            print(f"  → 文件2数据更一致，标准差小了 {((std1-std2)/std2*100):.1f}%")
            print(f"  → 文件1数据更多样，标准差是文件2的 {(std1/std2):.1f} 倍")
        else:
            print("! 两个文件的标准差相同")
        
        if abs(std2 - std1) > 0.1:
            print("✓ 两个数据集具有明显不同的一致性特征")
        else:
            print("! 两个数据集的一致性特征较为相似")

# 简化的单文件分析函数
def quick_analyze(file_path):
    """快速分析单个文件"""
    boxes = read_detection_data(file_path)
    if not boxes:
        print(f"无法读取文件: {file_path}")
        return None
    
    ratios = [w/h for w, h in boxes]
    std = np.std(ratios)
    mean = np.mean(ratios)
    
    print(f"文件: {file_path}")
    print(f"样本数: {len(boxes)}")
    print(f"宽高比: 均值={mean:.3f}, 标准差={std:.3f}")
    
    if std < 0.12:
        print("→ 数据集1类型 (一致性高的目标)")
    else:
        print("→ 数据集2类型 (多样性高的目标)")
    print("-" * 40)
    
    return std  # 返回标准差用于比较

def compare_two_files(file1_path, file2_path):
    """比较两个文件的标准差"""
    print("快速对比分析:")
    print("=" * 40)
    
    std1 = quick_analyze(file1_path)
    std2 = quick_analyze(file2_path)
    
    if std1 is not None and std2 is not None:
        print("标准差对比结果:")
        if std1 < std2:
            print(f"✓ {file1_path} 的标准差更小")
            print(f"  数值: {std1:.3f} < {std2:.3f}")
            print(f"  差异: {file2_path} 的标准差是 {file1_path} 的 {(std2/std1):.1f} 倍")
        elif std2 < std1:
            print(f"✓ {file2_path} 的标准差更小")
            print(f"  数值: {std2:.3f} < {std1:.3f}")
            print(f"  差异: {file1_path} 的标准差是 {file2_path} 的 {(std1/std2):.1f} 倍")
        else:
            print(f"! 两个文件的标准差相同: {std1:.3f}")
        print("=" * 40)

# 使用示例
if __name__ == "__main__":
    
    result_1 = "mot17.txt"
    # result_2 = "dance.txt"
    result_2 = "mot20.txt"
    
    # 方法1: 分析两个文件
    print("方法1: 完整分析")
    analyze_files(result_1, result_2)
    
    print("\n" + "="*60)
    print("方法2: 快速分析")
    
    # 方法2: 快速分析
    quick_analyze(result_1)
    quick_analyze(result_2)
    
    print("\n" + "="*60)
    print("方法3: 快速对比")
    
    # 方法3: 专门对比标准差
    compare_two_files(result_1, result_2)
    
    print("\n分类标准说明:")
    print("• 标准差 < 0.12: 数据集1类型 (一致性目标，如固定比例的人/车)")
    print("• 标准差 ≥ 0.12: 数据集2类型 (多样性目标，如多类别或多视角)")
    print("• 标准差越小，数据越一致；标准差越大，数据越多样化")