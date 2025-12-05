from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt

kf = KalmanFilter(dim_x=2, dim_z=1)
kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])  # 状态迁移矩阵
kf.H = np.array([[1.0, 0.0]])  # 观测矩阵，只观测位置
kf.P *= 1000.0  # 初始协方差
kf.R = 10  # 测量噪声协方差
from filterpy.common import Q_discrete_white_noise

# kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=0.1)  # 过程噪声协方差
# print(kf.Q)
kf.Q = np.array([[0.01, 0.0], [0.0, 0.01]])  # 过程噪声协方差
measurements = [0.97, 0.97, 0.98, 0.97, 0.97, 0.97, 0.97, 0.96, 0.96, 0.96, 0.96, 0.94, 0.88, 0.86, 0.85, 0.69, 0.48]
# print(len(measurements))  # 打印测量值的数量
kf.x = np.array([0.0, 0.97])  # 初始状态：位置0，速度0
results = []
for z in measurements:
    kf.predict()
    kf.update(z)
    results.append(kf.x[0])  # 记录位置估计
    # print(kf.x[0])  # 即每步的状态估计
# print(len(results))  # 打印位置估计的数量
# print(results)  # 打印所有位置估计结果


# 时间从420开始
time = np.arange(420, 420 + len(measurements))

# 绘图
# Plot
plt.figure(figsize=(10, 6))
plt.plot(time, measurements, marker="o", label="Raw Data", linewidth=2, linestyle="--")
plt.plot(time, results, marker="s", label="Kalman Filter", linewidth=2)

plt.xlabel("Time")
plt.ylabel("Confidence")
plt.title("Raw Data vs Kalman Filter")
plt.legend()
# plt.grid(True)

# Save figure
plt.savefig("kalman_confidence.png", dpi=300, bbox_inches="tight")
plt.close()

print("Plot saved to kalman_confidence.png")
