import numpy as np

def kalman_p_calculator(P, Q, R, num_iterations):
    """
    简化版Kalman滤波器P矩阵计算器
    
    Parameters:
    -----------
    P_init : list
        初始协方差矩阵对角线元素，8个元素
    Q : list  
        过程噪声协方差矩阵对角线元素，8个元素
    R : list
        观测噪声协方差矩阵对角线元素，4个元素
    num_iterations : int
        计算的迭代次数
        
    Returns:
    --------
    P_final : list
        最终的P矩阵对角线元素
    """
    
    F = np.array(
            [
                # x y w h x' y' w' h'
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
    
    H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )
    
    
    # 迭代计算
    for i in range(num_iterations):
            
        # 预测步骤: P_pred = F * P * F' + Q
        P_pred = F @ P @ F.T + Q
        
        # 更新步骤
        # S = H * P_pred * H' + R
        S = H @ P_pred @ H.T + R
        
        # K = P_pred * H' * inv(S)
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # P = (I - K*H) * P_pred * (I - K*H)' + K * R * K'
        I_KH = np.eye(8) - K @ H
        P = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
    
    return np.diag(P).tolist()
    

if __name__ == "__main__":
    P_init = np.diag([1, 1, 1, 1, 1000, 1000, 1000, 1000])
    Q = np.diag([1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01])
    R = np.diag([10, 30, 100, 500])
    
    print(kalman_p_calculator(P=P_init, Q=Q, R=R, num_iterations=4))