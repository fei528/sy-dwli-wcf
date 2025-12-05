import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class BBoxIDAnalyzer:
    def __init__(self, data_root):
        self.data_root = data_root
        self.datasets = {
            'DANCE-val': 'DANCE-val',
            'MOT17-val': 'MOT17-val',
            'MOT20-val': 'MOT20-val'
        }

    def parse_gt_txt_with_ids(self, gt_path):
        """解析gt.txt文件，按ID收集bbox数据"""
        id_data = defaultdict(lambda: {
            'widths': [],
            'heights': []
        })

        if not os.path.exists(gt_path):
            return id_data

        try:
            with open(gt_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(',')
                if len(parts) < 6:
                    continue

                try:
                    track_id = int(parts[1])
                    width = float(parts[4])
                    height = float(parts[5])

                    if width > 0 and height > 0:
                        aspect_ratio = width / height

                        # 过滤条件：宽度≤500，高度≤800，宽高比≤2.3
                        if width <= 500 and height <= 800 and aspect_ratio <= 2.3 and aspect_ratio >= 0.1:
                            id_data[track_id]['widths'].append(width)
                            id_data[track_id]['heights'].append(height)

                except (ValueError, IndexError):
                    continue

        except Exception as e:
            print(f"读取 {gt_path} 时出错: {e}")

        return dict(id_data)

    def calculate_dimension_change_percentages(self):
        """计算每个ID的宽度和高度变化百分比"""
        all_results = {}

        for dataset_name, dataset_folder in self.datasets.items():
            dataset_path = os.path.join(self.data_root, dataset_folder)

            if not os.path.exists(dataset_path):
                print(f"跳过数据集 {dataset_name}: 路径不存在")
                continue

            print(f"分析数据集: {dataset_name}")

            width_changes = []
            height_changes = []
            valid_id_count = 0

            # 调试信息
            debug_data = []

            # 获取所有序列路径
            for seq_name in sorted(os.listdir(dataset_path)):
                seq_path = os.path.join(dataset_path, seq_name)
                if os.path.isdir(seq_path):
                    gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
                    if os.path.exists(gt_path):
                        seq_id_data = self.parse_gt_txt_with_ids(gt_path)

                        for track_id, data in seq_id_data.items():
                            # 只分析至少有5帧数据的轨迹
                            if len(data['widths']) >= 5:
                                widths = np.array(data['widths'])
                                heights = np.array(data['heights'])

                                # 计算变化百分比：(max - min) / min * 100
                                width_min, width_max = widths.min(), widths.max()
                                height_min, height_max = heights.min(), heights.max()

                                # 分别计算宽度和高度的变化百分比
                                width_change_pct = None
                                height_change_pct = None

                                if width_min > 0:
                                    width_change_pct = (
                                        width_max - width_min) / width_min * 100

                                if height_min > 0:
                                    height_change_pct = (
                                        height_max - height_min) / height_min * 100

                                # 只有当两个维度都计算成功时才计入统计
                                if width_change_pct is not None and height_change_pct is not None:
                                    width_changes.append(width_change_pct)
                                    height_changes.append(height_change_pct)
                                    valid_id_count += 1

                                    # 收集调试数据
                                    debug_data.append({
                                        'seq': seq_name,
                                        'id': track_id,
                                        'width_min': width_min,
                                        'width_max': width_max,
                                        'width_change': width_change_pct,
                                        'height_min': height_min,
                                        'height_max': height_max,
                                        'height_change': height_change_pct
                                    })

                        print(
                            f"  处理 {seq_name}: {len(seq_id_data)} 个ID", end='\r')

            print()

            # 调试输出前几个样本
            if debug_data:
                print(f"  调试信息 - 前5个ID的变化:")
                for i, item in enumerate(debug_data[:5]):
                    print(
                        f"    ID {item['id']} ({item['seq']}): W={item['width_change']:.1f}%, H={item['height_change']:.1f}%")

                # 检查是否所有宽度和高度变化都相同
                unique_width_changes = len(set(width_changes))
                unique_height_changes = len(set(height_changes))
                print(f"  宽度变化的唯一值数量: {unique_width_changes}")
                print(f"  高度变化的唯一值数量: {unique_height_changes}")

                # 检查是否宽度和高度变化列表完全相同
                if len(width_changes) == len(height_changes):
                    width_array = np.array(width_changes)
                    height_array = np.array(height_changes)
                    if np.array_equal(width_array, height_array):
                        print(f"  警告: 宽度和高度变化列表完全相同!")
                    else:
                        print(f"  宽度和高度变化列表不同 (这是正常的)")

            all_results[dataset_name] = {
                'width_changes': width_changes,
                'height_changes': height_changes,
                'valid_id_count': valid_id_count
            }

            print(f"  有效ID数量: {valid_id_count}")
            if width_changes:
                print(
                    f"  宽度变化范围: {min(width_changes):.1f}% - {max(width_changes):.1f}%")
                print(
                    f"  宽度变化均值: {np.mean(width_changes):.1f}% ± {np.std(width_changes):.1f}%")
            if height_changes:
                print(
                    f"  高度变化范围: {min(height_changes):.1f}% - {max(height_changes):.1f}%")
                print(
                    f"  高度变化均值: {np.mean(height_changes):.1f}% ± {np.std(height_changes):.1f}%")

        return all_results

    def get_global_range(self, results):
        """计算所有数据的全局范围，用于统一刻度"""
        all_width_changes = []
        all_height_changes = []

        for dataset_name, data in results.items():
            if data['width_changes']:
                all_width_changes.extend(data['width_changes'])
            if data['height_changes']:
                all_height_changes.extend(data['height_changes'])

        # 计算全局范围
        if all_width_changes and all_height_changes:
            all_changes = all_width_changes + all_height_changes
            global_min = min(all_changes)
            global_max = min(max(all_changes), 1000)  # 限制最大值为1000

            # 添加一些边距
            margin = (global_max - global_min) * 0.05
            global_min = max(0, global_min - margin)  # 确保最小值不小于0
            global_max = min(global_max + margin, 1000)  # 确保不超过1000

            return global_min, global_max

        return 0, 1000  # 默认范围，最大值设为1000

    def create_change_percentage_plots(self, results):
        """创建变化百分比分布图 - 使用统一刻度"""
        datasets = ['DANCE-val', 'MOT17-val', 'MOT20-val']
        colors = ['#E74C3C', '#2ECC71', '#3498DB']

        # 获取全局范围
        global_min, global_max = self.get_global_range(results)
        print(f"全局刻度范围: {global_min:.1f}% - {global_max:.1f}%")

        # 创建1x2的子图布局
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 宽度变化分布
        ax_width = axes[0]
        for i, dataset in enumerate(datasets):
            if dataset in results and results[dataset]['width_changes']:
                changes = results[dataset]['width_changes']

                # 绘制直方图
                mean_val = np.mean(changes)
                ax_width.hist(changes, bins=30, alpha=0.6, color=colors[i],
                              label=f'{dataset} (n={len(changes)}, μ={mean_val:.1f}%)',
                              density=True, range=(global_min, global_max))

                # 添加均值线
                ax_width.axvline(mean_val, color=colors[i], linestyle='--',
                                 linewidth=2, alpha=0.8)

        ax_width.set_xlabel('Width Change Percentage (%)', fontsize=14)
        ax_width.set_ylabel('Density', fontsize=14)
        ax_width.set_title('Distribution of Width Change Percentages by ID',
                           fontsize=16, fontweight='bold')
        ax_width.legend(fontsize=12)
        ax_width.grid(True, alpha=0.3)
        ax_width.set_xlim(global_min, global_max)

        # 高度变化分布
        ax_height = axes[1]
        for i, dataset in enumerate(datasets):
            if dataset in results and results[dataset]['height_changes']:
                changes = results[dataset]['height_changes']

                # 绘制直方图
                mean_val = np.mean(changes)
                ax_height.hist(changes, bins=30, alpha=0.6, color=colors[i],
                               label=f'{dataset} (n={len(changes)}, μ={mean_val:.1f}%)',
                               density=True, range=(global_min, global_max))

                # 添加均值线
                ax_height.axvline(mean_val, color=colors[i], linestyle='--',
                                  linewidth=2, alpha=0.8)

        ax_height.set_xlabel('Height Change Percentage (%)', fontsize=14)
        ax_height.set_ylabel('Density', fontsize=14)
        ax_height.set_title('Distribution of Height Change Percentages by ID',
                            fontsize=16, fontweight='bold')
        ax_height.legend(fontsize=12)
        ax_height.grid(True, alpha=0.3)
        ax_height.set_xlim(global_min, global_max)

        plt.tight_layout()
        plt.savefig('id_dimension_change_percentages.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("ID维度变化百分比分布图已保存: id_dimension_change_percentages.png")

        # 保存宽度子图
        self.save_width_subplot(results, global_min,
                                global_max, datasets, colors)

        # 保存高度子图
        self.save_height_subplot(
            results, global_min, global_max, datasets, colors)

        # 创建分离的对比图
        self.create_separated_comparison_plots(results, global_min, global_max)

    def save_width_subplot(self, results, global_min, global_max, datasets, colors):
        """单独保存宽度变化分布图"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for i, dataset in enumerate(datasets):
            if dataset in results and results[dataset]['width_changes']:
                changes = results[dataset]['width_changes']

                # 绘制直方图
                mean_val = np.mean(changes)
                ax.hist(changes, bins=30, alpha=0.6, color=colors[i],
                        label=f'{dataset} (n={len(changes)}, μ={mean_val:.1f}%)',
                        density=True, range=(global_min, global_max))

                # 添加均值线
                ax.axvline(mean_val, color=colors[i], linestyle='--',
                           linewidth=2, alpha=0.8)

        ax.set_xlabel('Width Change Percentage (%)', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(global_min, global_max)

        plt.tight_layout()
        plt.savefig('width_change_distribution.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("宽度变化分布子图已保存: width_change_distribution.png")

    def save_height_subplot(self, results, global_min, global_max, datasets, colors):
        """单独保存高度变化分布图"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for i, dataset in enumerate(datasets):
            if dataset in results and results[dataset]['height_changes']:
                changes = results[dataset]['height_changes']

                # 绘制直方图
                mean_val = np.mean(changes)
                ax.hist(changes, bins=30, alpha=0.6, color=colors[i],
                        label=f'{dataset} (n={len(changes)}, μ={mean_val:.1f}%)',
                        density=True, range=(global_min, global_max))

                # 添加均值线
                ax.axvline(mean_val, color=colors[i], linestyle='--',
                           linewidth=2, alpha=0.8)

        ax.set_xlabel('Height Change Percentage (%)', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(global_min, global_max)

        plt.tight_layout()
        plt.savefig('height_change_distribution.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("高度变化分布子图已保存: height_change_distribution.png")

    def create_separated_comparison_plots(self, results, global_min, global_max):
        """创建分离的对比图，每个数据集单独显示 - 使用统一刻度"""
        datasets = ['DANCE-val', 'MOT17-val', 'MOT20-val']
        colors = ['#E74C3C', '#2ECC71', '#3498DB']

        fig, axes = plt.subplots(3, 2, figsize=(16, 18))

        for i, dataset in enumerate(datasets):
            # 宽度子图
            ax_w = axes[i, 0]
            # 高度子图
            ax_h = axes[i, 1]

            if dataset in results and results[dataset]['width_changes'] and results[dataset]['height_changes']:
                width_changes = results[dataset]['width_changes']
                height_changes = results[dataset]['height_changes']

                # 绘制宽度直方图
                ax_w.hist(width_changes, bins=40, alpha=0.7, color=colors[i],
                          density=True, edgecolor='black', linewidth=0.5,
                          range=(global_min, global_max))
                mean_w = np.mean(width_changes)
                ax_w.axvline(mean_w, color='red', linestyle='--', linewidth=3)
                ax_w.text(0.7, 0.9, f'Mean = {mean_w:.1f}%\nStd = {np.std(width_changes):.1f}%\nn = {len(width_changes)}',
                          transform=ax_w.transAxes, fontsize=12,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

                # 绘制高度直方图
                ax_h.hist(height_changes, bins=40, alpha=0.7, color=colors[i],
                          density=True, edgecolor='black', linewidth=0.5,
                          range=(global_min, global_max))
                mean_h = np.mean(height_changes)
                ax_h.axvline(mean_h, color='red', linestyle='--', linewidth=3)
                ax_h.text(0.7, 0.9, f'Mean = {mean_h:.1f}%\nStd = {np.std(height_changes):.1f}%\nn = {len(height_changes)}',
                          transform=ax_h.transAxes, fontsize=12,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            else:
                # 如果没有数据，显示空图表但保持格式
                ax_w.text(0.5, 0.5, 'No Data Available', transform=ax_w.transAxes,
                          fontsize=16, ha='center', va='center',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
                ax_h.text(0.5, 0.5, 'No Data Available', transform=ax_h.transAxes,
                          fontsize=16, ha='center', va='center',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

            # 设置标题和标签（无论是否有数据都要设置）
            ax_w.set_title(f'{dataset} - Width Change Distribution',
                           fontsize=14, fontweight='bold')
            ax_w.set_xlabel('Width Change Percentage (%)', fontsize=12)
            ax_w.set_ylabel('Density', fontsize=12)
            ax_w.grid(True, alpha=0.3)
            ax_w.set_xlim(global_min, global_max)

            ax_h.set_title(
                f'{dataset} - Height Change Distribution', fontsize=14, fontweight='bold')
            ax_h.set_xlabel('Height Change Percentage (%)', fontsize=12)
            ax_h.set_ylabel('Density', fontsize=12)
            ax_h.grid(True, alpha=0.3)
            ax_h.set_xlim(global_min, global_max)

        plt.tight_layout()
        plt.savefig('id_dimension_change_separated.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("分离式ID维度变化图已保存: id_dimension_change_separated.png")

    def print_change_statistics(self, results):
        """打印变化百分比统计信息"""
        print("\n" + "="*60)
        print("ID维度变化百分比统计")
        print("变化百分比 = (max - min) / min × 100%")
        print("="*60)

        for dataset_name, data in results.items():
            if not data['width_changes'] and not data['height_changes']:
                continue

            print(f"\n{dataset_name}:")
            print("-" * 40)
            print(f"有效ID数量: {data['valid_id_count']}")

            if data['width_changes']:
                width_changes = np.array(data['width_changes'])
                print(f"\n宽度变化百分比:")
                print(
                    f"  均值: {np.mean(width_changes):.2f}% ± {np.std(width_changes):.2f}%")
                print(f"  中位数: {np.median(width_changes):.2f}%")
                print(
                    f"  范围: [{np.min(width_changes):.1f}%, {np.max(width_changes):.1f}%]")
                print(
                    f"  25%-75%分位数: [{np.percentile(width_changes, 25):.1f}%, {np.percentile(width_changes, 75):.1f}%]")
                print(f"  90%分位数: {np.percentile(width_changes, 90):.1f}%")

                # 统计不同变化程度的ID比例
                small_change = np.sum(width_changes <= 10) / \
                    len(width_changes) * 100
                medium_change = np.sum((width_changes > 10) & (
                    width_changes <= 50)) / len(width_changes) * 100
                large_change = np.sum(width_changes > 50) / \
                    len(width_changes) * 100

                print(f"  变化≤10%的ID: {small_change:.1f}%")
                print(f"  变化10-50%的ID: {medium_change:.1f}%")
                print(f"  变化>50%的ID: {large_change:.1f}%")

            if data['height_changes']:
                height_changes = np.array(data['height_changes'])
                print(f"\n高度变化百分比:")
                print(
                    f"  均值: {np.mean(height_changes):.2f}% ± {np.std(height_changes):.2f}%")
                print(f"  中位数: {np.median(height_changes):.2f}%")
                print(
                    f"  范围: [{np.min(height_changes):.1f}%, {np.max(height_changes):.1f}%]")
                print(
                    f"  25%-75%分位数: [{np.percentile(height_changes, 25):.1f}%, {np.percentile(height_changes, 75):.1f}%]")
                print(f"  90%分位数: {np.percentile(height_changes, 90):.1f}%")

                # 统计不同变化程度的ID比例
                small_change = np.sum(height_changes <= 10) / \
                    len(height_changes) * 100
                medium_change = np.sum((height_changes > 10) & (
                    height_changes <= 50)) / len(height_changes) * 100
                large_change = np.sum(height_changes > 50) / \
                    len(height_changes) * 100

                print(f"  变化≤10%的ID: {small_change:.1f}%")
                print(f"  变化10-50%的ID: {medium_change:.1f}%")
                print(f"  变化>50%的ID: {large_change:.1f}%")


def main():
    data_root = "/workspace/Deep-OC-SORT/results/gt"

    analyzer = BBoxIDAnalyzer(data_root)

    print("开始分析每个ID的宽度和高度变化百分比...")
    print("过滤条件: 宽度≤500, 高度≤800, 宽高比≤2.3, 轨迹长度≥5帧")

    # 计算变化百分比
    results = analyzer.calculate_dimension_change_percentages()

    if not results:
        print("未找到有效数据！")
        return

    # 创建分布图
    analyzer.create_change_percentage_plots(results)

    # 打印统计信息
    analyzer.print_change_statistics(results)

    print("\n分析完成！")
    print("生成的图表文件:")
    print("- id_dimension_change_percentages.png (变化百分比分布 - 组合图)")
    print("- width_change_distribution.png (宽度变化分布 - 单独子图)")
    print("- height_change_distribution.png (高度变化分布 - 单独子图)")
    print("- id_dimension_change_separated.png (分离式变化分布)")


if __name__ == "__main__":
    main()
