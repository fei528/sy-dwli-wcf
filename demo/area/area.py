import numpy as np
import pandas as pd
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse


class MOTAreaAnalyzer:
    """
    MOT GT文件检测框面积分析器
    支持分析每帧的检测框总面积和覆盖率
    """

    def __init__(self, image_width: int = 1920, image_height: int = 1080):
        """
        初始化分析器

        Args:
            image_width: 图像宽度
            image_height: 图像高度
        """
        self.image_width = image_width
        self.image_height = image_height
        self.total_image_area = image_width * image_height

    def read_mot_gt_file(self, gt_file_path: str) -> Dict[int, List[List[float]]]:
        """
        读取MOT格式的GT文件

        Args:
            gt_file_path: GT文件路径

        Returns:
            frame_boxes: {frame_id: [[x, y, w, h], ...]}
        """
        if not os.path.exists(gt_file_path):
            raise FileNotFoundError(f"GT文件不存在: {gt_file_path}")

        print(f"Reading GT file: {gt_file_path}")

        # MOT format: frame_id, object_id, x, y, w, h, conf, class_id, visibility
        try:
            df = pd.read_csv(gt_file_path, header=None)
            expected_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 9 columns

            if df.shape[1] < 6:
                raise ValueError(f"Invalid GT file format, at least 6 columns required, got {df.shape[1]} columns")

        except Exception as e:
            print(f"Error reading GT file: {e}")
            print("Trying other separators...")

            # Try other possible separators
            for sep in [" ", "\t", ";"]:
                try:
                    df = pd.read_csv(gt_file_path, header=None, sep=sep)
                    if df.shape[1] >= 6:
                        print(f"Successfully read file using separator '{sep}'")
                        break
                except:
                    continue
            else:
                raise ValueError("Cannot parse GT file, please check file format")

        # 提取需要的列: frame_id, x, y, w, h
        frame_id_col = df.iloc[:, 0].astype(int)
        x_col = df.iloc[:, 2].astype(float)
        y_col = df.iloc[:, 3].astype(float)
        w_col = df.iloc[:, 4].astype(float)
        h_col = df.iloc[:, 5].astype(float)

        # 过滤无效的检测框
        valid_mask = (w_col > 0) & (h_col > 0)
        if hasattr(df, "shape") and df.shape[1] >= 7:
            # 如果有confidence列，过滤掉置信度为0的
            conf_col = df.iloc[:, 6].astype(float)
            valid_mask = valid_mask & (conf_col > 0)

        frame_ids = frame_id_col[valid_mask]
        x_vals = x_col[valid_mask]
        y_vals = y_col[valid_mask]
        w_vals = w_col[valid_mask]
        h_vals = h_col[valid_mask]

        # 按帧ID分组
        frame_boxes = defaultdict(list)
        for frame_id, x, y, w, h in zip(frame_ids, x_vals, y_vals, w_vals, h_vals):
            # 确保坐标在图像范围内
            x = max(0, min(x, self.image_width - 1))
            y = max(0, min(y, self.image_height - 1))
            w = min(w, self.image_width - x)
            h = min(h, self.image_height - y)

            if w > 0 and h > 0:
                frame_boxes[frame_id].append([x, y, w, h])

        print(
            f"Successfully read {len(frame_boxes)} frames with total {sum(len(boxes) for boxes in frame_boxes.values())} detection boxes"
        )
        return dict(frame_boxes)

    def convert_xywh_to_xyxy(self, boxes: List[List[float]]) -> np.ndarray:
        """
        将XYWH格式转换为XYXY格式（优化版本）

        Args:
            boxes: [[x, y, w, h], ...]

        Returns:
            xyxy_boxes: numpy array of [[x1, y1, x2, y2], ...]
        """
        if not boxes:
            return np.array([]).reshape(0, 4)

        boxes_array = np.array(boxes)
        xyxy_boxes = np.zeros_like(boxes_array)

        xyxy_boxes[:, 0] = boxes_array[:, 0]  # x1 = x
        xyxy_boxes[:, 1] = boxes_array[:, 1]  # y1 = y
        xyxy_boxes[:, 2] = boxes_array[:, 0] + boxes_array[:, 2]  # x2 = x + w
        xyxy_boxes[:, 3] = boxes_array[:, 1] + boxes_array[:, 3]  # y2 = y + h

        return xyxy_boxes

    def calculate_union_area_optimized(self, boxes: List[List[float]]) -> float:
        """
        优化的扫描线算法计算并集面积

        Args:
            boxes: [[x, y, w, h], ...] 格式的检测框

        Returns:
            total_area: 去除重叠后的总面积
        """
        if not boxes:
            return 0.0

        # 转换为xyxy格式
        xyxy_boxes = self.convert_xywh_to_xyxy(boxes)

        # 收集所有事件
        events = []
        for i, (x1, y1, x2, y2) in enumerate(xyxy_boxes):
            if x2 > x1 and y2 > y1:  # 有效检测框
                events.append((x1, "start", y1, y2))
                events.append((x2, "end", y1, y2))

        if not events:
            return 0.0

        # 按x坐标排序
        events.sort()

        total_area = 0.0
        active_intervals = []
        prev_x = events[0][0]

        for x, event_type, y1, y2 in events:
            if x > prev_x and active_intervals:
                # 计算当前x区间的面积
                width = x - prev_x
                height = self._calculate_union_length_optimized(active_intervals)
                total_area += width * height

            # 更新活跃区间
            if event_type == "start":
                active_intervals.append((y1, y2))
            else:
                # 使用更高效的删除方法
                try:
                    active_intervals.remove((y1, y2))
                except ValueError:
                    pass  # 如果区间不存在，忽略

            prev_x = x

        return total_area

    def _calculate_union_length_optimized(self, intervals: List[Tuple[float, float]]) -> float:
        """
        优化的一维区间并集长度计算

        Args:
            intervals: [(start, end), ...]

        Returns:
            union_length: 并集长度
        """
        if not intervals:
            return 0.0

        # 排序并合并重叠区间
        intervals_sorted = sorted(intervals)
        merged = [intervals_sorted[0]]

        for start, end in intervals_sorted[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:  # 重叠或相邻
                merged[-1] = (last_start, max(last_end, end))
            else:  # 不重叠
                merged.append((start, end))

        # 计算总长度
        return sum(end - start for start, end in merged)

    def analyze_frame_areas(self, frame_boxes: Dict[int, List[List[float]]]) -> Dict[int, Dict]:
        """
        分析每帧的检测框面积

        Args:
            frame_boxes: {frame_id: [[x, y, w, h], ...]}

        Returns:
            frame_results: {frame_id: {'area': float, 'coverage': float, 'count': int, 'time': float}}
        """
        print("Starting frame area analysis...")

        frame_results = {}
        total_frames = len(frame_boxes)

        for i, (frame_id, boxes) in enumerate(frame_boxes.items()):
            # 显示进度
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Progress: {i + 1}/{total_frames} ({(i + 1)/total_frames*100:.1f}%)")

            # 计算面积和耗时
            start_time = time.perf_counter()
            area = self.calculate_union_area_optimized(boxes)
            end_time = time.perf_counter()

            # 计算覆盖率
            coverage_ratio = area / self.total_image_area

            frame_results[frame_id] = {
                "area": area,
                "coverage": coverage_ratio,
                "count": len(boxes),
                "time": end_time - start_time,
                "area_percentage": coverage_ratio * 100,  # 添加百分比格式
            }

        print(f"Analysis completed! Processed {total_frames} frames")
        return frame_results

    def print_statistics(self, frame_results: Dict[int, Dict]):
        """
        打印统计信息

        Args:
            frame_results: 帧分析结果
        """
        if not frame_results:
            print("No data to analyze")
            return

        areas = [result["area"] for result in frame_results.values()]
        coverages = [result["coverage"] for result in frame_results.values()]
        counts = [result["count"] for result in frame_results.values()]
        times = [result["time"] for result in frame_results.values()]

        print("\n" + "=" * 50)
        print("STATISTICS REPORT")
        print("=" * 50)

        print(f"Total frames: {len(frame_results)}")
        print(f"Image size: {self.image_width} x {self.image_height}")
        print(f"Total image area: {self.total_image_area:,}")

        print(f"\nDetection box count statistics:")
        print(f"  Average per frame: {np.mean(counts):.1f} boxes")
        print(
            f"  Minimum: {min(counts)} boxes (frame {min(frame_results.keys(), key=lambda k: frame_results[k]['count'])})"
        )
        print(
            f"  Maximum: {max(counts)} boxes (frame {max(frame_results.keys(), key=lambda k: frame_results[k]['count'])})"
        )

        print(f"\nArea statistics:")
        print(f"  Average area: {np.mean(areas):,.0f} pixels ({np.mean(coverages):.2%} of image)")
        print(f"  Minimum area: {min(areas):,.0f} pixels ({min(coverages):.2%} of image)")
        print(f"  Maximum area: {max(areas):,.0f} pixels ({max(coverages):.2%} of image)")
        print(f"  Standard deviation: {np.std(areas):,.0f} ({np.std(coverages):.2%})")

        print(f"\nCoverage ratio statistics:")
        print(f"  Average coverage: {np.mean(coverages):.2%}")
        print(f"  Minimum coverage: {min(coverages):.2%}")
        print(f"  Maximum coverage: {max(coverages):.2%}")
        print(f"  Coverage std deviation: {np.std(coverages):.2%}")

        # 添加覆盖率分布统计
        coverage_ranges = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]
        print(f"\nCoverage distribution:")
        for low, high in coverage_ranges:
            count = sum(1 for c in coverages if low <= c < high)
            percentage = count / len(coverages) * 100
            print(f"  {low:.1%} - {high:.1%}: {count} frames ({percentage:.1f}%)")

        print(f"\nPerformance statistics:")
        print(f"  Average computation time: {np.mean(times)*1000:.3f} ms/frame")
        print(f"  Total computation time: {sum(times):.3f} seconds")
        print(
            f"  Slowest frame: {max(times)*1000:.3f} ms (frame {max(frame_results.keys(), key=lambda k: frame_results[k]['time'])})"
        )

    def save_results(self, frame_results: Dict[int, Dict], output_path: str):
        """
        保存结果到CSV文件

        Args:
            frame_results: 帧分析结果
            output_path: 输出文件路径
        """
        # 转换为DataFrame
        data = []
        for frame_id, result in frame_results.items():
            data.append(
                {
                    "frame_id": frame_id,
                    "area_pixels": result["area"],
                    "area_percentage": result["coverage"] * 100,  # 转换为百分比
                    "coverage_ratio": result["coverage"],
                    "box_count": result["count"],
                    "compute_time_ms": result["time"] * 1000,
                }
            )

        df = pd.DataFrame(data)
        df = df.sort_values("frame_id")

        # 保存到CSV
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

        return df

    def visualize_frame_areas(
        self,
        frame_boxes: Dict[int, List[List[float]]],
        save_dir: str = "results/frame_visualizations",
        max_frames: int = 50,
    ):
        """
        为每一帧创建面积可视化图

        Args:
            frame_boxes: {frame_id: [[x, y, w, h], ...]}
            save_dir: 保存目录
            max_frames: 最大可视化帧数（避免生成过多图片）
        """
        import matplotlib

        matplotlib.use("Agg")

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 选择要可视化的帧（如果帧数太多，均匀采样）
        frame_ids = sorted(frame_boxes.keys())
        if len(frame_ids) > max_frames:
            step = len(frame_ids) // max_frames
            selected_frames = frame_ids[::step][:max_frames]
            print(f"Too many frames ({len(frame_ids)}), visualizing {len(selected_frames)} frames")
        else:
            selected_frames = frame_ids
            print(f"Visualizing all {len(selected_frames)} frames")

        # 为每一帧创建可视化
        for i, frame_id in enumerate(selected_frames):
            if (i + 1) % 10 == 0:
                print(f"Creating frame visualizations: {i + 1}/{len(selected_frames)}")

            boxes = frame_boxes[frame_id]
            self._create_single_frame_visualization(frame_id, boxes, save_dir)

        print(f"Frame visualizations saved to: {save_dir}")

    def _create_single_frame_visualization(self, frame_id: int, boxes: List[List[float]], save_dir: str):
        """
        创建单帧的面积可视化

        Args:
            frame_id: 帧ID
            boxes: 检测框列表
            save_dir: 保存目录
        """
        # 创建图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Frame {frame_id} - Detection Box Areas", fontsize=14, fontweight="bold")

        # 左图：检测框可视化
        ax1.set_xlim(0, self.image_width)
        ax1.set_ylim(self.image_height, 0)  # 翻转Y轴以匹配图像坐标
        ax1.set_aspect("equal")
        ax1.set_title(f"Detection Boxes (Count: {len(boxes)})")
        ax1.set_xlabel("X (pixels)")
        ax1.set_ylabel("Y (pixels)")

        # 绘制每个检测框
        total_area = 0
        colors = plt.cm.Set3(np.linspace(0, 1, len(boxes)))  # 为每个框分配不同颜色

        for i, (x, y, w, h) in enumerate(boxes):
            # 绘制矩形
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=colors[i], facecolor=colors[i], alpha=0.3)
            ax1.add_patch(rect)

            # 添加面积标签
            area = w * h
            total_area += area
            ax1.text(x + w / 2, y + h / 2, f"{area:.0f}", ha="center", va="center", fontsize=8, fontweight="bold")

        # 添加网格
        ax1.grid(True, alpha=0.3)

        # 右图：面积统计
        if boxes:
            individual_areas = [w * h for x, y, w, h in boxes]
            union_area = self.calculate_union_area_optimized(boxes)
            overlap_area = sum(individual_areas) - union_area
            coverage_ratio = union_area / self.total_image_area

            # 饼图显示面积组成
            if overlap_area > 0:
                labels = ["Union Area", "Overlap Area", "Background"]
                sizes = [union_area - overlap_area, overlap_area, self.total_image_area - union_area]
                colors_pie = ["lightblue", "orange", "lightgray"]
            else:
                labels = ["Union Area", "Background"]
                sizes = [union_area, self.total_image_area - union_area]
                colors_pie = ["lightblue", "lightgray"]

            # 创建饼图
            wedges, texts, autotexts = ax2.pie(
                sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%", startangle=90
            )
            ax2.set_title(f"Area Composition\nCoverage: {coverage_ratio:.2%}")

            # 添加详细统计文本
            stats_text = f"""Statistics:
Total Boxes: {len(boxes)}
Individual Sum: {sum(individual_areas):,.0f} px
Union Area: {union_area:,.0f} px
Overlap Area: {overlap_area:,.0f} px
Coverage: {coverage_ratio:.2%}
Avg Box Size: {np.mean(individual_areas):,.0f} px"""

            ax2.text(
                1.3,
                0.5,
                stats_text,
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )
        else:
            ax2.text(
                0.5,
                0.5,
                "No detection boxes in this frame",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=12,
            )
            ax2.set_title("No Data")

        plt.tight_layout()

        # 保存图片
        save_path = os.path.join(save_dir, f"frame_{frame_id:06d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def create_area_summary_chart(self, frame_results: Dict[int, Dict], save_path: str = None):
        """
        创建面积汇总图表

        Args:
            frame_results: 帧分析结果
            save_path: 保存路径
        """
        import matplotlib

        matplotlib.use("Agg")

        frame_ids = sorted(frame_results.keys())
        areas = [frame_results[fid]["area"] for fid in frame_ids]
        coverages = [frame_results[fid]["coverage"] * 100 for fid in frame_ids]
        counts = [frame_results[fid]["count"] for fid in frame_ids]

        # 创建大图表
        fig = plt.figure(figsize=(20, 12))

        # 主要面积趋势图（大图）
        ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=4, rowspan=2)

        # 双Y轴：面积和覆盖率
        ax_area = ax_main
        ax_coverage = ax_area.twinx()

        # 绘制面积
        line1 = ax_area.plot(frame_ids, areas, "b-", linewidth=1.5, alpha=0.8, label="Area (pixels)")
        ax_area.fill_between(frame_ids, areas, alpha=0.3, color="blue")
        ax_area.set_xlabel("Frame ID")
        ax_area.set_ylabel("Area (pixels)", color="blue")
        ax_area.tick_params(axis="y", labelcolor="blue")
        ax_area.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        # 绘制覆盖率
        line2 = ax_coverage.plot(frame_ids, coverages, "r-", linewidth=1.5, alpha=0.8, label="Coverage (%)")
        ax_coverage.set_ylabel("Coverage (%)", color="red")
        ax_coverage.tick_params(axis="y", labelcolor="red")

        # 标题和网格
        ax_main.set_title("Detection Box Area and Coverage Trends", fontsize=16, fontweight="bold", pad=20)
        ax_area.grid(True, alpha=0.3)

        # 图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_main.legend(lines, labels, loc="upper left")

        # 子图1：面积分布直方图
        ax1 = plt.subplot2grid((3, 4), (2, 0))
        ax1.hist(areas, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        ax1.set_title("Area Distribution")
        ax1.set_xlabel("Area (pixels)")
        ax1.set_ylabel("Frequency")
        ax1.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))

        # 子图2：覆盖率分布直方图
        ax2 = plt.subplot2grid((3, 4), (2, 1))
        ax2.hist(coverages, bins=20, alpha=0.7, color="lightgreen", edgecolor="black")
        ax2.set_title("Coverage Distribution")
        ax2.set_xlabel("Coverage (%)")
        ax2.set_ylabel("Frequency")

        # 子图3：检测框数量
        ax3 = plt.subplot2grid((3, 4), (2, 2))
        ax3.plot(frame_ids, counts, "g-", linewidth=1, alpha=0.7)
        ax3.fill_between(frame_ids, counts, alpha=0.3, color="green")
        ax3.set_title("Box Count per Frame")
        ax3.set_xlabel("Frame ID")
        ax3.set_ylabel("Count")

        # 子图4：面积vs检测框数量散点图
        ax4 = plt.subplot2grid((3, 4), (2, 3))
        scatter = ax4.scatter(counts, coverages, alpha=0.6, c=frame_ids, cmap="viridis", s=20)
        ax4.set_title("Coverage vs Box Count")
        ax4.set_xlabel("Box Count")
        ax4.set_ylabel("Coverage (%)")
        plt.colorbar(scatter, ax=ax4, label="Frame ID")

        plt.tight_layout()

        # 保存
        if not save_path:
            save_path = "results/area_summary_detailed.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Detailed area summary chart saved to: {save_path}")
        plt.close()

    def plot_results(self, frame_results: Dict[int, Dict], save_path: Optional[str] = None):
        """
        绘制分析结果图表（原有功能保持不变）
        """
        import matplotlib

        matplotlib.use("Agg")

        frame_ids = sorted(frame_results.keys())
        areas = [frame_results[fid]["area"] for fid in frame_ids]
        coverages = [frame_results[fid]["coverage"] * 100 for fid in frame_ids]
        counts = [frame_results[fid]["count"] for fid in frame_ids]
        times = [frame_results[fid]["time"] * 1000 for fid in frame_ids]

        # 创建子图 - 增加一个子图显示面积占比
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle("MOT GT Detection Box Area Analysis", fontsize=16, fontweight="bold")

        # 面积变化
        ax1.plot(frame_ids, areas, "b-", linewidth=1, alpha=0.7)
        ax1.set_title("Total Detection Box Area per Frame")
        ax1.set_xlabel("Frame ID")
        ax1.set_ylabel("Area (pixels)")
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        # 覆盖率变化
        ax2.plot(frame_ids, coverages, "g-", linewidth=1, alpha=0.7)
        ax2.set_title("Image Coverage Ratio per Frame")
        ax2.set_xlabel("Frame ID")
        ax2.set_ylabel("Coverage (%)")
        ax2.grid(True, alpha=0.3)

        # 检测框数量
        ax3.plot(frame_ids, counts, "r-", linewidth=1, alpha=0.7)
        ax3.set_title("Number of Detection Boxes per Frame")
        ax3.set_xlabel("Frame ID")
        ax3.set_ylabel("Box Count")
        ax3.grid(True, alpha=0.3)

        # 计算时间
        ax4.plot(frame_ids, times, "m-", linewidth=1, alpha=0.7)
        ax4.set_title("Computation Time per Frame")
        ax4.set_xlabel("Frame ID")
        ax4.set_ylabel("Time (ms)")
        ax4.grid(True, alpha=0.3)

        # 面积占比分布直方图
        ax5.hist(coverages, bins=20, alpha=0.7, color="orange", edgecolor="black")
        ax5.set_title("Coverage Ratio Distribution")
        ax5.set_xlabel("Coverage (%)")
        ax5.set_ylabel("Number of Frames")
        ax5.grid(True, alpha=0.3)

        # 面积vs检测框数量散点图
        scatter = ax6.scatter(counts, coverages, alpha=0.6, c=frame_ids, cmap="viridis", s=20)
        ax6.set_title("Coverage vs Box Count")
        ax6.set_xlabel("Number of Boxes")
        ax6.set_ylabel("Coverage (%)")
        ax6.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax6, label="Frame ID")

        plt.tight_layout()

        # 如果没有指定保存路径，使用默认路径
        if not save_path:
            # 创建results目录
            os.makedirs("results", exist_ok=True)
            save_path = "results/area_analysis.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {save_path}")
        plt.close()  # 关闭图形，释放内存


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MOT GT Detection Box Area Analyzer")
    parser.add_argument("gt_file", help="GT文件路径")
    parser.add_argument("--width", type=int, default=1920, help="Image width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Image height (default: 1080)")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    parser.add_argument("--plot", help="Save chart file path")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate charts")
    parser.add_argument("--visualize-frames", action="store_true", help="Create individual frame visualizations")
    parser.add_argument("--max-frames", type=int, default=50, help="Maximum frames to visualize (default: 50)")
    parser.add_argument(
        "--frame-vis-dir", default="results/frame_visualizations", help="Directory for frame visualizations"
    )

    args = parser.parse_args()

    # 创建分析器
    analyzer = MOTAreaAnalyzer(args.width, args.height)

    try:
        # 读取GT文件
        frame_boxes = analyzer.read_mot_gt_file(args.gt_file)

        # 分析每帧面积
        frame_results = analyzer.analyze_frame_areas(frame_boxes)

        # 打印统计信息
        analyzer.print_statistics(frame_results)

        # 保存结果
        if args.output:
            # 确保输出目录存在
            output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "results"
            os.makedirs(output_dir, exist_ok=True)
            analyzer.save_results(frame_results, args.output)
        else:
            # 默认保存到results目录
            os.makedirs("results", exist_ok=True)
            analyzer.save_results(frame_results, "results/area_analysis.csv")

        # 绘制图表
        if not args.no_plot:
            if args.plot:
                plot_dir = os.path.dirname(args.plot) if os.path.dirname(args.plot) else "results"
                os.makedirs(plot_dir, exist_ok=True)
                analyzer.plot_results(frame_results, args.plot)
            else:
                # 默认保存到results目录
                os.makedirs("results", exist_ok=True)
                analyzer.plot_results(frame_results, "results/area_analysis.png")

            # 创建详细汇总图表
            analyzer.create_area_summary_chart(frame_results, "results/area_summary_detailed.png")

        # 创建每帧可视化
        if args.visualize_frames:
            analyzer.visualize_frame_areas(frame_boxes, args.frame_vis_dir, args.max_frames)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


# 使用示例
if __name__ == "__main__":
    # 如果作为脚本运行，使用命令行参数
    import sys

    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        # 示例用法
        print("MOT GT Detection Box Area Analyzer")
        print("\nUsage:")
        print("python script.py <gt_file_path> [options]")
        print("\nOptions:")
        print("  --width WIDTH     Image width (default: 1920)")
        print("  --height HEIGHT   Image height (default: 1080)")
        print("  --output PATH     Save CSV results file")
        print("  --plot PATH       Save chart file")
        print("  --no-plot         Don't generate charts")
        print("  --visualize-frames Create individual frame visualizations")
        print("  --max-frames N    Maximum frames to visualize (default: 50)")
        print("  --frame-vis-dir DIR Directory for frame visualizations")
        print("\nExample:")
        print("python script.py gt.txt --width 1920 --height 1080 --output results.csv --plot chart.png")
        print("python script.py gt.txt --visualize-frames --max-frames 100")
