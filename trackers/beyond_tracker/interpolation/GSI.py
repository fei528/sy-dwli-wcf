"""
@Author: Du Yunhao
@Filename: GSI.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/1 9:18
@Discription: Gaussian-smoothed interpolation
"""
import os
import numpy as np
import argparse
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

# 线性插值
def LinearInterpolation(input_, interval):
    input_ = input_[np.lexsort([input_[:, 0], input_[:, 1]])]  # 按ID和帧排序
    output_ = input_.copy()
    '''线性插值'''
    id_pre, f_pre, row_pre = -1, -1, np.zeros((10,))
    for row in input_:
        f_curr, id_curr = row[:2].astype(int)
        if id_curr == id_pre:  # 同ID
            if f_pre + 1 < f_curr < f_pre + interval:
                for i, f in enumerate(range(f_pre + 1, f_curr), start=1):  # 逐框插值
                    step = (row - row_pre) / (f_curr - f_pre) * i
                    row_new = row_pre + step
                    output_ = np.append(output_, row_new[np.newaxis, :], axis=0)
        else:  # 不同ID
            id_pre = id_curr
        row_pre = row
        f_pre = f_curr
    output_ = output_[np.lexsort([output_[:, 0], output_[:, 1]])]
    return output_

# 高斯平滑
def GaussianSmooth(input_, tau):
    output_ = list()
    ids = set(input_[:, 1])
    for id_ in ids:
        tracks = input_[input_[:, 1] == id_]
        len_scale = np.clip(tau * np.log(tau ** 3 / len(tracks)), tau ** -1, tau ** 2)
        gpr = GPR(RBF(len_scale, 'fixed'))
        t = tracks[:, 0].reshape(-1, 1)
        x = tracks[:, 2].reshape(-1, 1)
        y = tracks[:, 3].reshape(-1, 1)
        w = tracks[:, 4].reshape(-1, 1)
        h = tracks[:, 5].reshape(-1, 1)
        gpr.fit(t, x)
        xx = gpr.predict(t)
        gpr.fit(t, y)
        yy = gpr.predict(t)
        gpr.fit(t, w)
        ww = gpr.predict(t)
        gpr.fit(t, h)
        hh = gpr.predict(t)
        output_.extend([
            [t[i, 0], id_, xx[i], yy[i], ww[i], hh[i], 1, -1, -1 , -1] for i in range(len(t))
        ])
    return output_


def GSInterpolation(dir_in, dir_out, interval, tau):
    # 确保输出目录存在
    os.makedirs(dir_out, exist_ok=True)

    # 遍历输入目录下所有txt文件
    for filename in os.listdir(dir_in):
        if filename.endswith('.txt'):
            path_in = os.path.join(dir_in, filename)
            path_out = os.path.join(dir_out, filename)

            # 读取文件
            input_ = np.loadtxt(path_in, delimiter=',')
            # 线性插值
            li = LinearInterpolation(input_, interval)
            # 高斯平滑
            gsi = GaussianSmooth(li, tau)
            # 保存结果
            np.savetxt(path_out, gsi, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')



# if __name__ == "__main__":
#     input_dir = "/workspace/Deep-OC-SORT/results/trackers/MOT17-val/new_kf/data"
#     output_dir = "/workspace/Deep-OC-SORT/results/trackers/MOT17-val/new_kf_gpr/data"
    
#     # 间隔参数和平滑度参数，可根据需要调整
#     interval = 30  # 与原代码保持一致
#     tau= 10 # 平滑因子，值越大越平滑，0表示精确插值
    
#     GSInterpolation(input_dir, output_dir, interval, tau=tau)

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='GS轨迹平滑处理')
    
    # 添加命令行参数
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='输入目录路径 (例如: /path/to/MOT17-val/new_kf)')
    parser.add_argument('--interval', type=int, default=30, help='间隔参数，默认为30')
    parser.add_argument('--tau', type=float, default=10, help='平滑因子，值越大越平滑，默认为10')
    
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
    output_dir_name = f"{dir_name}_gs"
    
    # 构建完整的输出路径
    output_dir = os.path.join(parent_dir, output_dir_name, "data")
    
    interval = args.interval
    tau = args.tau
    
    print(f"输入目录基路径: {base_input_dir}")
    print(f"实际处理的输入路径: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"间隔参数: {interval}")
    print(f"平滑因子: {tau}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    
    # 执行GBRC轨迹平滑
    GSInterpolation(input_dir, output_dir, interval, tau=tau)