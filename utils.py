import glob
import os

import numpy as np
import shutil
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF


def write_results_no_score(filename, results):
    """最终优化版本"""
    # 预处理：收集所有有效数据
    valid_data = []
    print(results)
    for frame_id, tlwhs, track_ids in results:
        for tlwh, track_id in zip(tlwhs, track_ids):
            if track_id >= 0:  # 只收集有效数据
                x1, y1, w, h = tlwh
                valid_data.append((frame_id, track_id, x1, y1, w, h))

    # 批量格式化
    lines = [
        f"{frame},{track_id},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},-1,-1,-1,-1"
        for frame, track_id, x1, y1, w, h in valid_data
    ]

    # 一次性写入
    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_results(filename, results):
    """最终优化版本"""
    # 预处理：收集所有有效数据
    valid_data = []
    for frame_id, tlwhs, track_ids in results:
        for tlwh, track_id in zip(tlwhs, track_ids):
            if track_id >= 0:  # 只收集有效数据
                x1, y1, w, h = tlwh
                valid_data.append((frame_id, track_id, x1, y1, w, h))

    # 批量格式化
    lines = [
        f"{frame},{track_id},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},-1,-1,-1,-1"
        for frame, track_id, x1, y1, w, h in valid_data
    ]

    # 一次性写入
    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")


def filter_targets(online_targets, aspect_ratio_thresh, min_box_area):
    """Removes targets not meeting threshold criteria.

    Returns (list of tlwh, list of ids).
    """
    online_tlwhs = []
    online_ids = []
    for t in online_targets:
        tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
        tid = t[4]
        vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
        if tlwh[2] * tlwh[3] > min_box_area and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
    return online_tlwhs, online_ids


# 线性插值
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


# 高斯插值
def dti_with_gpr(txt_path, save_path, n_min=30, n_dti=20, tau=1.5):
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

    def apply_gpr(input_, tau):
        output_ = []
        ids = set(input_[:, 1])  # 所有不同的目标ID
        for id_ in ids:
            tracks = input_[input_[:, 1] == id_]
            if len(tracks) < 2:
                output_.extend(tracks.tolist())  # 如果只有一帧，不插值
                continue

            # 动态调整核函数的尺度参数
            len_scale = np.clip(tau * np.log(tau**3 / len(tracks)), tau**-1, tau**2)

            # 初始化GPR
            gpr = GPR(kernel=RBF(len_scale, length_scale_bounds="fixed"))

            # 帧号作为输入，bbox各项为输出
            t = tracks[:, 0].reshape(-1, 1)  # 帧号
            x = tracks[:, 2]  # x1
            y = tracks[:, 3]  # y1
            w = tracks[:, 4]  # w
            h = tracks[:, 5]  # h

            # 对每一维坐标分别拟合与预测
            gpr.fit(t, x)
            xx = gpr.predict(t)

            gpr.fit(t, y)
            yy = gpr.predict(t)

            gpr.fit(t, w)
            ww = gpr.predict(t)

            gpr.fit(t, h)
            hh = gpr.predict(t)

            # 组合预测结果
            smoothed = np.zeros((len(t), 10), dtype=np.float64)
            smoothed[:, 0] = t[:, 0]  # 帧号
            smoothed[:, 1] = id_  # track id
            smoothed[:, 2] = xx  # x1
            smoothed[:, 3] = yy  # y1
            smoothed[:, 4] = ww  # w
            smoothed[:, 5] = hh  # h
            smoothed[:, 6] = 1  # score
            smoothed[:, 7:] = -1  # 其他字段占位

            output_.extend(smoothed.tolist())

        return np.array(output_)

    # 读取所有 txt 文件
    seq_txts = sorted(glob.glob(os.path.join(txt_path, "*.txt")))
    for seq_txt in seq_txts:
        seq_name = seq_txt.replace("\\", "/").split("/")[-1]
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

                    # 插值逻辑
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

                # 构建插值数据
                num_dti = len(frames_dti.keys())
                if num_dti > 0:
                    data_dti = np.zeros((num_dti, 10), dtype=np.float64)
                    for n in range(num_dti):
                        data_dti[n, 0] = list(frames_dti.keys())[n]
                        data_dti[n, 1] = track_id
                        data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
                        data_dti[n, 6:] = [1, -1, -1, -1]
                    tracklet_dti = np.vstack((tracklet, data_dti))

            # 应用高斯平滑
            tracklet_dti = apply_gpr(tracklet_dti, tau)

            # 合并入总轨迹
            seq_results = np.vstack((seq_results, tracklet_dti))

        # 保存结果
        save_seq_txt = os.path.join(save_path, seq_name)
        seq_results = seq_results[1:]  # 去掉初始化的第一行
        seq_results = seq_results[seq_results[:, 0].argsort()]
        dti_write_results(save_seq_txt, seq_results)


if __name__ == "__main__":
    post_folder = "results/trackers/MOT17-val/1122_final_test_post"
    pre_folder = "results/trackers/MOT17-val/1122_final_test"
    if os.path.exists(post_folder):
        print(f"Overwriting previous results in {post_folder}")
        shutil.rmtree(post_folder)
    shutil.copytree(pre_folder, post_folder)
    post_folder_data = os.path.join(post_folder, "data")
    dti(post_folder_data, post_folder_data)
    print(f"Linear interpolation post-processing applied, saved to {post_folder_data}.")
