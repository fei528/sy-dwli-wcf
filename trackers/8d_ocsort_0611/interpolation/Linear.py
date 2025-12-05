import numpy as np
import glob
import os
import argparse

def dti(txt_path, save_path, n_min=30, n_dti=20):
    def dti_write_results(filename, results):
        save_format = "{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n"
        with open(filename, "w") as f:
            for i in range(results.shape[0]):
                frame_data = results[i]
                frame_id = int(frame_data[0])
                track_id = int(frame_data[1])
                x1, y1, w, h = frame_data[2:6]
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=-1)
                f.write(line)
    seq_txts = sorted(glob.glob(os.path.join(txt_path, "*.txt")))
    # breakpoint()
    for seq_txt in seq_txts:
        seq_name = seq_txt.replace("\\", "/").split("/")[-1]  ## To better play along with windows paths
        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=",")
        min_id = int(np.min(seq_data[:, 1]))
        max_id = int(np.max(seq_data[:, 1]))
        seq_results = np.zeros((1, 10), dtype=np.float64)
        for track_id in range(min_id, max_id + 1):
            index = seq_data[:, 1] == track_id
            tracklet = seq_data[index]
            tracklet_dti = tracklet
            if tracklet.shape[0] == 0:
                continue
            n_frame = tracklet.shape[0]
            n_conf = np.sum(tracklet[:, 6] > 0.5)
            if n_frame > n_min:
                frames = tracklet[:, 0]
                frames_dti = {}
                for i in range(0, n_frame):
                    right_frame = frames[i]
                    if i > 0:
                        left_frame = frames[i - 1]
                    else:
                        left_frame = frames[i]
                    # disconnected track interpolation
                    if 1 < right_frame - left_frame < n_dti:
                        num_bi = int(right_frame - left_frame - 1)
                        right_bbox = tracklet[i, 2:6]
                        left_bbox = tracklet[i - 1, 2:6]
                        for j in range(1, num_bi + 1):
                            curr_frame = j + left_frame
                            curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / (
                                right_frame - left_frame
                            ) + left_bbox
                            frames_dti[curr_frame] = curr_bbox
                num_dti = len(frames_dti.keys())
                if num_dti > 0:
                    data_dti = np.zeros((num_dti, 10), dtype=np.float64)
                    for n in range(num_dti):
                        data_dti[n, 0] = list(frames_dti.keys())[n]
                        data_dti[n, 1] = track_id
                        data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
                        data_dti[n, 6:] = [1, -1, -1, -1]
                    tracklet_dti = np.vstack((tracklet, data_dti))
            seq_results = np.vstack((seq_results, tracklet_dti))
        save_seq_txt = os.path.join(save_path, seq_name)
        seq_results = seq_results[1:]
        seq_results = seq_results[seq_results[:, 0].argsort()]
        dti_write_results(save_seq_txt, seq_results)


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='GS轨迹平滑处理')
    
    # 添加命令行参数
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='输入目录路径 (例如: /path/to/MOT17-val/new_kf)')
    
    # 解析参数
    args = parser.parse_args()
    
    # 获取输入基础路径
    base_input_dir = args.input_dir
    
    # 构建完整的输入路径（添加data子目录）
    input_dir = os.path.join(base_input_dir, "data")
    
    # 构建输出路径
    # 从输入路径中提取目录名
    dir_name = os.path.basename(base_input_dir)
    parent_dir = os.path.dirname(base_input_dir)
    
    # 创建输出目录名（添加_gs后缀）
    output_dir_name = f"{dir_name}_linear"
    
    # 构建完整的输出路径
    output_dir = os.path.join(parent_dir, output_dir_name, "data")
    
    print(f"输入目录基路径: {base_input_dir}")
    print(f"实际处理的输入路径: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 确保输出目录存在
    # os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)  # 修改这里

    
    # 执行GBRC轨迹平滑
    dti(input_dir, output_dir)