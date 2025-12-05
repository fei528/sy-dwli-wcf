import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict


class BBoxAnalyzer:
    def __init__(self, data_root):
        self.data_root = data_root
        self.datasets = {
            'DANCE-val': 'DANCE-val',
            'MOT17-val': 'MOT17-val',
            'MOT20-val': 'MOT20-val'
        }

    def parse_gt_txt(self, gt_path):
        """解析gt.txt文件，收集所有bbox数据"""
        bbox_data = {
            'widths': [],
            'heights': [],
            'areas': [],
            'aspect_ratios': []
        }

        if not os.path.exists(gt_path):
            return bbox_data

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
                    width = float(parts[4])
                    height = float(parts[5])

                    if width > 0 and height > 0:
                        aspect_ratio = width / height

                        # 过滤条件：宽度≤500，高度≤800，宽高比≤2.3
                        if width <= 500 and height <= 800 and aspect_ratio <= 2.3 and aspect_ratio >= 0.1:
                            area = width * height
                            bbox_data['widths'].append(width)
                            bbox_data['heights'].append(height)
                            bbox_data['areas'].append(area)
                            bbox_data['aspect_ratios'].append(aspect_ratio)

                except (ValueError, IndexError):
                    continue

        except Exception as e:
            print(f"读取 {gt_path} 时出错: {e}")

        return bbox_data

    def analyze_datasets(self):
        """分析所有数据集"""
        distribution_data = {}

        for dataset_name, dataset_folder in self.datasets.items():
            dataset_path = os.path.join(self.data_root, dataset_folder)

            if not os.path.exists(dataset_path):
                print(f"跳过数据集 {dataset_name}: 路径不存在")
                continue

            print(f"分析数据集: {dataset_name}")

            # 收集所有bbox数据
            all_widths = []
            all_heights = []
            all_areas = []
            all_aspect_ratios = []

            # 获取所有序列路径
            seq_paths = []
            for seq_name in sorted(os.listdir(dataset_path)):
                seq_path = os.path.join(dataset_path, seq_name)
                if os.path.isdir(seq_path):
                    gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
                    if os.path.exists(gt_path):
                        seq_paths.append((seq_name, gt_path))

            # 处理序列
            for seq_name, gt_path in seq_paths:
                bbox_data = self.parse_gt_txt(gt_path)
                print(
                    f"  处理 {seq_name}: {len(bbox_data['widths'])} 个框", end='\r')

                all_widths.extend(bbox_data['widths'])
                all_heights.extend(bbox_data['heights'])
                all_areas.extend(bbox_data['areas'])
                all_aspect_ratios.extend(bbox_data['aspect_ratios'])

            print()  # 换行

            if all_widths:
                # 对面积过滤异常值（保留5%-95%分位数）
                areas_array = np.array(all_areas)
                p5, p95 = np.percentile(areas_array, [5, 95])
                area_mask = (areas_array >= p5) & (areas_array <= p95)
                areas_filtered = areas_array[area_mask]

                distribution_data[dataset_name] = {
                    'widths': all_widths,
                    'heights': all_heights,
                    'areas': all_areas,
                    'areas_filtered': areas_filtered.tolist(),
                    'aspect_ratios': all_aspect_ratios
                }

                print(f"  总框数量: {len(all_widths)}")
                print(f"  过滤条件: 宽度≤500, 高度≤800, 宽高比≤2.3")
                print(f"  面积过滤前: {len(all_areas)}，过滤后: {len(areas_filtered)}")
                print(f"  面积范围: [{min(all_areas):.0f}, {max(all_areas):.0f}]")
                print(
                    f"  过滤后面积范围: [{areas_filtered.min():.0f}, {areas_filtered.max():.0f}]")

        return distribution_data

    def create_bbox_distribution_plots(self, distribution_data):
        """创建所有bbox的分布图"""
        datasets = ['DANCE-val', 'MOT17-val', 'MOT20-val']
        colors = ['#E74C3C', '#2ECC71', '#3498DB']

        # 创建2x2的子图布局，增大图片尺寸
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        dimensions = [
            ('widths', 'Width Distribution'),
            ('heights', 'Height Distribution'),
            ('areas_filtered', 'Area Distribution (Filtered)'),
            ('aspect_ratios', 'Aspect Ratio Distribution')
        ]

        # 子图标记和标题（中文+英文）
        subplot_info = [
            ('a)', 'BBox宽度分布', 'Width Distribution of BBox'),
            ('b)', 'BBox高度分布', 'Height Distribution of BBox'),
            ('c)', 'BBox面积分布（过滤后）', 'Area Distribution of BBox (Filtered)'),
            ('d)', 'BBox宽高比分布', 'Aspect Ratio Distribution of BBox')
        ]

        for dim_idx, (dimension, title) in enumerate(dimensions):
            row = dim_idx // 2
            col = dim_idx % 2
            ax = axes[row, col]

            for i, dataset in enumerate(datasets):
                if dataset in distribution_data:
                    values = np.array(distribution_data[dataset][dimension])

                    if len(values) > 1:
                        print(
                            f"{dataset} {dimension}: {len(values)} values, range [{values.min():.1f}, {values.max():.1f}]")

                        # 对于大数据集，适度采样提高绘图速度
                        if len(values) > 50000:
                            sample_indices = np.random.choice(
                                len(values), 30000, replace=False)
                            values_for_kde = values[sample_indices]
                        else:
                            values_for_kde = values

                        # 使用KDE绘制平滑分布
                        density = stats.gaussian_kde(values_for_kde)

                        # 创建x轴范围
                        x_min, x_max = values.min(), values.max()
                        xs = np.linspace(x_min, x_max, 300)
                        ys = density(xs)

                        # 计算标签
                        mean_val = np.mean(values)
                        if dimension in ['areas', 'areas_filtered'] and mean_val > 1000:
                            label = f'{dataset} (μ={mean_val:.0f})'
                        elif dimension == 'aspect_ratios':
                            label = f'{dataset} (μ={mean_val:.2f})'
                        else:
                            label = f'{dataset} (μ={mean_val:.1f})'

                        ax.plot(xs, ys, label=label,
                                color=colors[i], linewidth=3.0)  # 增加线宽
                        ax.fill_between(xs, ys, alpha=0.3, color=colors[i])

            # 设置标签
            if dimension in ['widths', 'heights']:
                xlabel = 'Pixels'
            elif dimension in ['areas', 'areas_filtered']:
                xlabel = 'Area (pixels²)'
            else:
                xlabel = 'Aspect Ratio'

            # 获取子图标记和标题
            subplot_label, chinese_title, english_title = subplot_info[dim_idx]

            # 增大字体大小
            ax.set_ylabel('Density', fontsize=16, fontweight='normal')
            ax.set_xlabel(xlabel, fontsize=16, fontweight='normal')

            # 设置双行标题格式：子图标记 + 中文标题 + 英文标题
            ax.set_title(f'{subplot_label} {chinese_title}\n{english_title}',
                         fontsize=16, fontweight='bold', pad=20)

            # 增大刻度标签字体
            ax.tick_params(axis='both', which='major', labelsize=14)

            # 增大图例字体
            ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)

        plt.tight_layout(pad=2.0)  # 增加子图间距
        plt.savefig('bbox_dimension_distribution.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("BBox尺寸分布图表已保存: bbox_dimension_distribution.png")

        # 保存各个子图为独立文件
        self._save_individual_plots(distribution_data)

    def create_area_zoom_plot(self, distribution_data):
        """创建面积分布的局部放大图"""
        datasets = ['DANCE-val', 'MOT17-val', 'MOT20-val']
        colors = ['#E74C3C', '#2ECC71', '#3498DB']

        # 创建1x2的子图布局，增大尺寸
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # 存储KDE信息用于局部放大
        kde_info = {}

        # 左图：完整面积分布（过滤后）
        ax_full = axes[0]
        for i, dataset in enumerate(datasets):
            if dataset in distribution_data:
                values = np.array(distribution_data[dataset]['areas_filtered'])

                if len(values) > 1:
                    # 采样处理大数据集
                    if len(values) > 50000:
                        sample_indices = np.random.choice(
                            len(values), 30000, replace=False)
                        values_for_kde = values[sample_indices]
                    else:
                        values_for_kde = values

                    density = stats.gaussian_kde(values_for_kde)
                    xs = np.linspace(values.min(), values.max(), 300)
                    ys = density(xs)

                    # 保存KDE信息
                    kde_info[dataset] = {
                        'density': density,
                        'values': values,
                        'color': colors[i]
                    }

                    mean_val = np.mean(values)
                    label = f'{dataset} (μ={mean_val:.0f})'

                    ax_full.plot(xs, ys, label=label,
                                 color=colors[i], linewidth=3.0)
                    ax_full.fill_between(xs, ys, alpha=0.3, color=colors[i])

        ax_full.set_xlabel('Area (pixels²)', fontsize=16, fontweight='normal')
        ax_full.set_ylabel('Density', fontsize=16, fontweight='normal')
        ax_full.set_title('a) BBox完整面积分布（过滤后）\nFull Area Distribution of BBox (Filtered)',
                          fontsize=16, fontweight='bold', pad=20)
        ax_full.tick_params(axis='both', which='major', labelsize=14)
        ax_full.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
        ax_full.grid(True, alpha=0.3)

        # 右图：局部放大 (0-25000)
        ax_zoom = axes[1]
        for dataset, info in kde_info.items():
            density = info['density']
            values = info['values']
            color = info['color']

            # 使用相同的KDE在0-25000范围评估
            xs_zoom = np.linspace(0, 25000, 500)
            ys_zoom = density(xs_zoom)

            # 计算0-25000范围内的统计
            zoom_values = values[(values >= 0) & (values <= 25000)]
            if len(zoom_values) > 0:
                mean_val = np.mean(zoom_values)
                label = f'{dataset} (μ={mean_val:.0f}, n={len(zoom_values)})'
            else:
                label = f'{dataset} (no data in range)'

            ax_zoom.plot(xs_zoom, ys_zoom, label=label,
                         color=color, linewidth=3.0)
            ax_zoom.fill_between(xs_zoom, ys_zoom, alpha=0.3, color=color)

        ax_zoom.set_xlabel('Area (pixels²)', fontsize=16, fontweight='normal')
        ax_zoom.set_ylabel('Density', fontsize=16, fontweight='normal')
        ax_zoom.set_title('b) BBox面积分布（局部放大：0-25000）\nArea Distribution of BBox (Zoom: 0-25000)',
                          fontsize=16, fontweight='bold', pad=20)
        ax_zoom.tick_params(axis='both', which='major', labelsize=14)
        ax_zoom.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.set_xlim(0, 25000)

        plt.tight_layout(pad=2.0)
        plt.savefig('area_distribution_zoom.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("面积分布局部放大图表已保存: area_distribution_zoom.png")

        # 保存面积分布的独立子图
        self._save_individual_area_plots(distribution_data)

    def _save_individual_plots(self, distribution_data):
        """保存各个维度的独立子图"""
        datasets = ['DANCE-val', 'MOT17-val', 'MOT20-val']
        colors = ['#E74C3C', '#2ECC71', '#3498DB']

        dimensions = [
            ('widths', 'Width Distribution'),
            ('heights', 'Height Distribution'),
            ('areas_filtered', 'Area Distribution (Filtered)'),
            ('aspect_ratios', 'Aspect Ratio Distribution')
        ]

        # 文件名
        filenames = [
            'width_distribution.png',
            'height_distribution.png',
            'area_distribution_filtered.png',
            'aspect_ratio_distribution.png'
        ]

        for dim_idx, (dimension, title) in enumerate(dimensions):
            # 创建单独的图
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            for i, dataset in enumerate(datasets):
                if dataset in distribution_data:
                    values = np.array(distribution_data[dataset][dimension])

                    if len(values) > 1:
                        # 对于大数据集，适度采样提高绘图速度
                        if len(values) > 50000:
                            sample_indices = np.random.choice(
                                len(values), 30000, replace=False)
                            values_for_kde = values[sample_indices]
                        else:
                            values_for_kde = values

                        # 使用KDE绘制平滑分布
                        density = stats.gaussian_kde(values_for_kde)

                        # 创建x轴范围
                        x_min, x_max = values.min(), values.max()
                        xs = np.linspace(x_min, x_max, 300)
                        ys = density(xs)

                        # 计算标签
                        mean_val = np.mean(values)
                        if dimension in ['areas', 'areas_filtered'] and mean_val > 1000:
                            label = f'{dataset} (μ={mean_val:.0f})'
                        elif dimension == 'aspect_ratios':
                            label = f'{dataset} (μ={mean_val:.2f})'
                        else:
                            label = f'{dataset} (μ={mean_val:.1f})'

                        ax.plot(xs, ys, label=label,
                                color=colors[i], linewidth=3.0)
                        ax.fill_between(xs, ys, alpha=0.3, color=colors[i])

            # 设置标签
            if dimension in ['widths', 'heights']:
                xlabel = 'Pixels'
            elif dimension in ['areas', 'areas_filtered']:
                xlabel = 'Area (pixels²)'
            else:
                xlabel = 'Aspect Ratio'

            # 设置标签（不设置标题）
            ax.set_ylabel('Density', fontsize=18, fontweight='normal')
            ax.set_xlabel(xlabel, fontsize=18, fontweight='normal')

            # 设置刻度和图例
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.legend(fontsize=16, frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)

            # 保存独立图片
            plt.tight_layout()
            plt.savefig(filenames[dim_idx], dpi=300, bbox_inches='tight')
            plt.close()

            print(f"独立子图已保存: {filenames[dim_idx]}")

    def _save_individual_area_plots(self, distribution_data):
        """保存面积分布的独立子图"""
        datasets = ['DANCE-val', 'MOT17-val', 'MOT20-val']
        colors = ['#E74C3C', '#2ECC71', '#3498DB']

        # 存储KDE信息
        kde_info = {}
        for i, dataset in enumerate(datasets):
            if dataset in distribution_data:
                values = np.array(distribution_data[dataset]['areas_filtered'])
                if len(values) > 1:
                    if len(values) > 50000:
                        sample_indices = np.random.choice(
                            len(values), 30000, replace=False)
                        values_for_kde = values[sample_indices]
                    else:
                        values_for_kde = values

                    density = stats.gaussian_kde(values_for_kde)
                    kde_info[dataset] = {
                        'density': density,
                        'values': values,
                        'color': colors[i]
                    }

        # 保存完整面积分布图（不要标题）
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        for dataset, info in kde_info.items():
            values = info['values']
            density = info['density']
            color = info['color']

            xs = np.linspace(values.min(), values.max(), 300)
            ys = density(xs)

            mean_val = np.mean(values)
            label = f'{dataset} (μ={mean_val:.0f})'

            ax.plot(xs, ys, label=label, color=color, linewidth=3.0)
            ax.fill_between(xs, ys, alpha=0.3, color=color)

        ax.set_xlabel('Area (pixels²)', fontsize=18, fontweight='normal')
        ax.set_ylabel('Density', fontsize=18, fontweight='normal')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=16, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('area_distribution_full.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("独立子图已保存: area_distribution_full.png")

        # 保存局部放大面积分布图（不要标题）
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        for dataset, info in kde_info.items():
            density = info['density']
            values = info['values']
            color = info['color']

            xs_zoom = np.linspace(0, 25000, 500)
            ys_zoom = density(xs_zoom)

            zoom_values = values[(values >= 0) & (values <= 25000)]
            if len(zoom_values) > 0:
                mean_val = np.mean(zoom_values)
                label = f'{dataset} (μ={mean_val:.0f}, n={len(zoom_values)})'
            else:
                label = f'{dataset} (no data in range)'

            ax.plot(xs_zoom, ys_zoom, label=label, color=color, linewidth=3.0)
            ax.fill_between(xs_zoom, ys_zoom, alpha=0.3, color=color)

        ax.set_xlabel('Area (pixels²)', fontsize=18, fontweight='normal')
        ax.set_ylabel('Density', fontsize=18, fontweight='normal')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=16, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 25000)

        plt.tight_layout()
        plt.savefig('area_distribution_zoom_individual.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("独立子图已保存: area_distribution_zoom_individual.png")

    def print_detailed_statistics(self, distribution_data):
        """打印详细统计信息"""
        print("\n" + "="*60)
        print("过滤后的BBox分布统计")
        print("过滤条件: 宽度≤500, 高度≤800, 宽高比≤2.3")
        print("="*60)

        for dataset in ['DANCE-val', 'MOT17-val', 'MOT20-val']:
            if dataset in distribution_data:
                print(f"\n{dataset}:")
                print("-" * 40)

                widths = np.array(distribution_data[dataset]['widths'])
                heights = np.array(distribution_data[dataset]['heights'])
                areas = np.array(distribution_data[dataset]['areas'])
                areas_filtered = np.array(
                    distribution_data[dataset]['areas_filtered'])
                aspect_ratios = np.array(
                    distribution_data[dataset]['aspect_ratios'])

                print(f"总框数量: {len(widths):,}")

                print(f"\n宽度分布:")
                print(f"  均值: {np.mean(widths):.2f} ± {np.std(widths):.2f}")
                print(f"  中位数: {np.median(widths):.2f}")
                print(f"  范围: [{np.min(widths):.1f}, {np.max(widths):.1f}]")
                print(
                    f"  四分位数: [{np.percentile(widths, 25):.1f}, {np.percentile(widths, 75):.1f}]")

                print(f"\n高度分布:")
                print(f"  均值: {np.mean(heights):.2f} ± {np.std(heights):.2f}")
                print(f"  中位数: {np.median(heights):.2f}")
                print(f"  范围: [{np.min(heights):.1f}, {np.max(heights):.1f}]")
                print(
                    f"  四分位数: [{np.percentile(heights, 25):.1f}, {np.percentile(heights, 75):.1f}]")

                print(f"\n面积分布:")
                print(
                    f"  原始数据 - 均值: {np.mean(areas):.0f} ± {np.std(areas):.0f}")
                print(
                    f"  原始数据 - 范围: [{np.min(areas):.0f}, {np.max(areas):.0f}]")
                print(
                    f"  过滤后 - 均值: {np.mean(areas_filtered):.0f} ± {np.std(areas_filtered):.0f}")
                print(
                    f"  过滤后 - 范围: [{np.min(areas_filtered):.0f}, {np.max(areas_filtered):.0f}]")
                print(f"  过滤后 - 中位数: {np.median(areas_filtered):.0f}")
                print(
                    f"  过滤后 - 四分位数: [{np.percentile(areas_filtered, 25):.0f}, {np.percentile(areas_filtered, 75):.0f}]")

                print(f"\n宽高比分布:")
                print(
                    f"  均值: {np.mean(aspect_ratios):.3f} ± {np.std(aspect_ratios):.3f}")
                print(f"  中位数: {np.median(aspect_ratios):.3f}")
                print(
                    f"  范围: [{np.min(aspect_ratios):.2f}, {np.max(aspect_ratios):.2f}]")
                print(
                    f"  四分位数: [{np.percentile(aspect_ratios, 25):.2f}, {np.percentile(aspect_ratios, 75):.2f}]")

                # 小面积统计
                small_areas = areas_filtered[areas_filtered <= 25000]
                print(f"\n小面积统计 (≤25000):")
                print(
                    f"  数量: {len(small_areas):,} ({len(small_areas)/len(areas_filtered)*100:.1f}%)")
                if len(small_areas) > 0:
                    print(f"  均值: {np.mean(small_areas):.0f}")
                    print(f"  中位数: {np.median(small_areas):.0f}")


def main():
    data_root = "/workspace/Deep-OC-SORT/results/gt"

    analyzer = BBoxAnalyzer(data_root)

    print("开始过滤后的BBox分布分析...")
    print("过滤条件: 宽度≤500, 高度≤800, 宽高比≤2.3")
    distribution_data = analyzer.analyze_datasets()

    if not distribution_data:
        print("未找到有效数据！")
        return

    # 创建bbox分布图
    analyzer.create_bbox_distribution_plots(distribution_data)

    # 创建面积分布的局部放大图
    analyzer.create_area_zoom_plot(distribution_data)

    # 打印详细统计信息
    analyzer.print_detailed_statistics(distribution_data)

    print("分析完成！")


if __name__ == "__main__":
    main()
