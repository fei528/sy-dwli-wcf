"""
The Gradient Boosting Reconnection Context (GBRC)
mechanism is developed to realize gradient-adaptive
reconnection of the fragment tracks with trajectory drifting noise
"""
import numpy as np
import os
import argparse
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.ensemble import GradientBoostingRegressor

def LinearInterpolation(input_, interval):
    """
    对轨迹点进行线性插值
    """
    input_ = input_[np.lexsort([input_[:, 0], input_[:, 1]])]
    output_ = input_.copy()
    id_pre, f_pre, row_pre = -1, -1, np.zeros((10,))
    for row in input_:
        f_curr, id_curr = row[:2].astype(int)
        if id_curr == id_pre:
            if f_pre + 1 < f_curr < f_pre + interval:
                for i, f in enumerate(range(f_pre + 1, f_curr), start=1):
                    step = (row - row_pre) / (f_curr - f_pre) * i
                    row_new = row_pre + step
                    output_ = np.append(output_, row_new[np.newaxis, :], axis=0)
        else:
            id_pre = id_curr
        row_pre = row
        f_pre = f_curr
    output_ = output_[np.lexsort([output_[:, 0], output_[:, 1]])]
    return output_

def GradientBoostingSmooth(input_, tau):
    """
    使用梯度提升回归(GBRC)进行轨迹平滑
    """
    output_ = list()
    ids = set(input_[:, 1])
    for id_ in ids:
        tracks = input_[input_[:, 1] == id_]
        
        # 如果轨迹点数太少，不进行平滑处理
        if len(tracks) <= 3:
            output_.extend([[t[0], id_, t[2], t[3], t[4], t[5], 1, -1, -1, -1] for t in tracks])
            continue
            
        len_scale = np.clip(tau * np.log(tau ** 3 / len(tracks)), tau ** -1, tau ** 2)
        gpr = GPR(RBF(len_scale, 'fixed'))
        
        # 提取特征和目标变量
        t = tracks[:, 0].reshape(-1, 1)  # 特征变量保持为列向量
        x = tracks[:, 2].reshape(-1)     # 目标变量转为一维数组
        y = tracks[:, 3].reshape(-1)     # 目标变量转为一维数组
        w = tracks[:, 4].reshape(-1)     # 目标变量转为一维数组
        h = tracks[:, 5].reshape(-1)     # 目标变量转为一维数组
        
        # 高斯过程回归
        try:
            gpr.fit(t, x)
            xx = gpr.predict(t)
            gpr.fit(t, y)
            yy = gpr.predict(t)
            gpr.fit(t, w)
            ww = gpr.predict(t)
            gpr.fit(t, h)
            hh = gpr.predict(t)
        except Exception as e:
            print(f"高斯过程回归出错：{e}")
            # 如果高斯过程回归失败，使用原始数据
            xx, yy, ww, hh = x, y, w, h
        
        # 梯度提升回归
        try:
            regr = GradientBoostingRegressor(n_estimators=115, learning_rate=0.065, min_samples_split=6)
            
            # 使用一维数组作为目标变量
            regr.fit(t, x)
            xx = regr.predict(t)
            
            regr.fit(t, y)
            yy = regr.predict(t)
            
            regr.fit(t, w)
            ww = regr.predict(t)
            
            regr.fit(t, h)
            hh = regr.predict(t)
        except Exception as e:
            print(f"梯度提升回归出错：{e}")
            # 如果梯度提升回归失败，保留之前的结果
            pass
        
        # 确保所有数组都是一维的
        t_flat = t.reshape(-1)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        ww = ww.reshape(-1)
        hh = hh.reshape(-1)
        
        output_.extend([
            [t_flat[i], id_, xx[i], yy[i], ww[i], hh[i], 1, -1, -1, -1] for i in range(len(t_flat))
        ])
    
    return np.array(output_)

def GBRCInterpolation(dir_in, dir_out, interval, tau):
    """
    处理目录中的所有轨迹文件，应用GBRC平滑
    
    参数:
        dir_in: 输入目录，包含轨迹文件
        dir_out: 输出目录，保存处理后的轨迹
        interval: 插值间隔
        tau: 平滑因子
    """
    # 确保输出目录存在
    os.makedirs(dir_out, exist_ok=True)
    
    # 获取输入目录中的所有txt文件
    files = [f for f in os.listdir(dir_in) if f.endswith('.txt')]
    
    # 统计文件数量以便报告进度
    total_files = len(files)
    print(f"找到 {total_files} 个轨迹文件需要处理")
    
    # 处理每个文件
    for i, filename in enumerate(files):
        path_in = os.path.join(dir_in, filename)
        path_out = os.path.join(dir_out, filename)
        
        try:
            # 读取输入文件
            input_ = np.loadtxt(path_in, delimiter=',')
            
            # 检查文件是否为空
            if len(input_) == 0 or input_.size == 0:
                print(f"警告: 空文件 {path_in}, 创建空输出文件")
                with open(path_out, 'w') as f:
                    pass
                continue
            
            # 确保输入数据是二维的
            if input_.ndim == 1:
                input_ = input_.reshape(1, -1)
            
            # 线性插值
            li = LinearInterpolation(input_, interval)
            
            # 梯度提升平滑
            gbrc = GradientBoostingSmooth(li, tau)
            
            # 保存结果
            np.savetxt(path_out, gbrc, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')
            
            # 报告进度
            print(f"处理完成 [{i+1}/{total_files}]: {filename}")
            
        except Exception as e:
            print(f"处理 {path_in} 时出错: {e}")
    
    print("所有文件处理完成")

# if __name__ == "__main__":
#     input_dir = "/workspace/Deep-OC-SORT/results/trackers/MOT17-val/new_kf/data"
#     output_dir = "/workspace/Deep-OC-SORT/results/trackers/MOT17-val/new_kf_gprc/data"
    
#     # 间隔参数和平滑度参数，可根据需要调整
#     interval = 30  # 与原代码保持一致
#     tau = 10       # 平滑因子，值越大越平滑
    
#     # 执行GBRC轨迹平滑
#     GBRCInterpolation(input_dir, output_dir, interval, tau=tau)
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='GBRC轨迹平滑处理')
    
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
    
    # 创建输出目录名（添加_gprc后缀）
    output_dir_name = f"{dir_name}_gprc"
    
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
    GBRCInterpolation(input_dir, output_dir, interval, tau=tau)