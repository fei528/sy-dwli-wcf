import numpy as np
import glob
import os
import argparse


def interpolate_bbox_refined(left_bbox, right_bbox, left_frame, right_frame, curr_frame, gap_size=None):
    """
    动态权重插值
    保持原始算法的核心，仅做微小改进
    """
    alpha = (curr_frame - left_frame) / (right_frame - left_frame)
    delta = right_bbox - left_bbox

    # 宽高变化量
    delta_w = abs(delta[2])
    delta_h = abs(delta[3])
    eps = 1e-6

    # 原始权重计算
    w_weight = delta_w / (delta_w + delta_h + eps)
    h_weight = delta_h / (delta_w + delta_h + eps)

    # 唯一的改进：对大gap进行权重衰减
    if gap_size and gap_size > 10:
        # gap越大，权重越趋向于0.5（更保守）
        decay_factor = min(1.0, 10.0 / gap_size)
        w_weight = 0.5 + (w_weight - 0.5) * decay_factor
        h_weight = 0.5 + (h_weight - 0.5) * decay_factor

    # 权重向量
    weights = np.array([1.0, 1.0, w_weight, h_weight])

    # 插值
    curr_bbox = left_bbox + weights * alpha * delta
    return curr_bbox


def DWLI(txt_path, save_path, n_min=30, n_dti=20, gap_threshold=None):
    """
    DTI主函数
    几乎保持原始代码，仅添加gap相关的微调
    """
    seq_txts = sorted(glob.glob(os.path.join(txt_path, "*.txt")))
    os.makedirs(save_path, exist_ok=True)

    total_files = len(seq_txts)
    print(f"开始处理 {total_files} 个文件...")
    if gap_threshold:
        print(f"使用gap阈值: {gap_threshold}")

    for file_idx, seq_txt in enumerate(seq_txts):
        seq_name = os.path.basename(seq_txt)
        print(f"\r处理进度: [{file_idx+1}/{total_files}] {seq_name}", end="", flush=True)

        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=",")
        min_id = int(np.min(seq_data[:, 1]))
        max_id = int(np.max(seq_data[:, 1]))
        seq_results = []

        for track_id in range(min_id, max_id + 1):
            index = seq_data[:, 1] == track_id
            tracklet = seq_data[index]
            if tracklet.shape[0] == 0:
                continue
            n_frame = tracklet.shape[0]
            if n_frame <= n_min:
                continue
            tracklet = tracklet[tracklet[:, 0].argsort()]
            frames = tracklet[:, 0]
            frames_dti = {}

            for i in range(1, n_frame):
                left_frame = int(frames[i - 1])
                right_frame = int(frames[i])
                gap = right_frame - left_frame

                # 使用gap阈值（如果提供）
                max_gap = gap_threshold if gap_threshold else n_dti

                if 1 < gap < max_gap:
                    left_bbox = tracklet[i - 1, 2:6]
                    right_bbox = tracklet[i, 2:6]
                    for j in range(1, gap):
                        curr_frame = left_frame + j
                        curr_bbox = interpolate_bbox_refined(
                            left_bbox, right_bbox, left_frame, right_frame, curr_frame, gap
                        )
                        frames_dti[curr_frame] = curr_bbox

            # 插值补上
            if frames_dti:
                data_dti = np.zeros((len(frames_dti), 10), dtype=np.float64)
                for idx, (f, box) in enumerate(frames_dti.items()):
                    data_dti[idx, 0] = f
                    data_dti[idx, 1] = track_id
                    data_dti[idx, 2:6] = box
                    data_dti[idx, 6:] = [1, -1, -1, -1]
                tracklet = np.vstack((tracklet, data_dti))
            seq_results.append(tracklet)

        # 合并保存
        if seq_results:
            all_results = np.vstack(seq_results)
            all_results = all_results[all_results[:, 0].argsort()]
            save_seq_txt = os.path.join(save_path, seq_name)
            dti_write_results(save_seq_txt, all_results)

    print(f"\n处理完成！共处理 {total_files} 个文件")


def dti_write_results(filename, results):
    save_format = "{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n"
    with open(filename, "w") as f:
        for row in results:
            frame_id, track_id, x1, y1, w, h = row[:6]
            line = save_format.format(frame=int(frame_id), id=int(track_id), x1=x1, y1=y1, w=w, h=h, s=-1)
            f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="异维多权线性插值")

    parser.add_argument("--input_dir", type=str, required=True, help="输入目录路径")
    parser.add_argument("--n_min", type=int, default=30, help="最小轨迹长度，默认30")
    parser.add_argument("--n_dti", type=int, default=20, help="最大插值间隔，默认20")
    parser.add_argument("--gap_threshold", type=int, default=None, help="自定义gap阈值，用于DanceTrack优化")

    args = parser.parse_args()

    base_input_dir = args.input_dir
    input_dir = os.path.join(base_input_dir, "data")

    dir_name = os.path.basename(base_input_dir)
    parent_dir = os.path.dirname(base_input_dir)
    output_dir_name = f"{dir_name}_dwli"
    output_dir = os.path.join(parent_dir, output_dir_name, "data")

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    DWLI(input_dir, output_dir, args.n_min, args.n_dti, args.gap_threshold)
