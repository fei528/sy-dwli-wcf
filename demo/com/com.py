import numpy as np

# 原始函数实现
def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx

def speed_direction_batch_lt(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:,0], dets[:,1]
    CX2, CY2 = tracks[:,0], tracks[:,1]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx

def speed_direction_batch_rt(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:,2], dets[:,1]
    CX2, CY2 = tracks[:,2], tracks[:,1]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx

def speed_direction_batch_lb(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:,0], dets[:,3]
    CX2, CY2 = tracks[:,0], tracks[:,3]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx

def speed_direction_batch_rb(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:,2], dets[:,3]
    CX2, CY2 = tracks[:,2], tracks[:,3]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx

def cost_vel(Y, X, trackers, velocities, detections, previous_obs, vdc_weight):
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores
    return angle_diff_cost

# 新的合并函数
def speed_direction_batch_all(dets, tracks):
    tracks = tracks[..., np.newaxis]
    
    # 定义所有关键点的坐标提取方式
    points = [
        # center: (x_min + x_max) / 2, (y_min + y_max) / 2
        ((dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0,
         (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0),
        # lt: x_min, y_min
        (dets[:, 0], dets[:, 1], tracks[:, 0], tracks[:, 1]),
        # rt: x_max, y_min
        (dets[:, 2], dets[:, 1], tracks[:, 2], tracks[:, 1]),
        # lb: x_min, y_max
        (dets[:, 0], dets[:, 3], tracks[:, 0], tracks[:, 3]),
        # rb: x_max, y_max
        (dets[:, 2], dets[:, 3], tracks[:, 2], tracks[:, 3])
    ]
    
    results = []
    for CX1, CY1, CX2, CY2 in points:
        dx = CX1 - CX2
        dy = CY1 - CY2
        norm = np.sqrt(dx**2 + dy**2) + 1e-6
        results.append(np.stack([dy / norm, dx / norm]))  # [2, M, N]
    
    return np.array(results)  # [5, 2, M, N]

def cost_vel_batch(directions, trackers, velocities, detections, previous_obs, vdc_weight):
    # 预计算公共部分
    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0
    valid_mask_expanded = np.repeat(valid_mask[:, np.newaxis], directions.shape[3], axis=1)  # [M, N]
    
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)  # [N, M]
    
    results = []
    # 处理所有五个点 [center=0, lt=1, rt=2, lb=3, rb=4]
    for i in range(5):
        Y, X = directions[i]  # [M, N]
        inertia_Y, inertia_X = velocities[:, i, 0], velocities[:, i, 1]  # [M]
        
        # 扩展到匹配形状
        inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)  # [M, N]
        inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)  # [M, N]
        
        # 计算角度差
        diff_angle_cos = inertia_X * X + inertia_Y * Y
        diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
        diff_angle = np.arccos(diff_angle_cos)
        diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi
        
        # 按照原代码逻辑计算成本
        angle_diff_cost = (valid_mask_expanded * diff_angle) * vdc_weight
        angle_diff_cost = angle_diff_cost.T  # [N, M]
        angle_diff_cost = angle_diff_cost * scores
        
        results.append(angle_diff_cost)
    
    return np.array(results)

# 测试案例
def test_cost_vel():
    # 创建测试数据
    np.random.seed(42)  # 固定随机种子
    
    # 3个检测框 [x_min, y_min, x_max, y_max, score]
    detections = np.array([
        [10, 20, 50, 60, 0.9],
        [15, 25, 55, 65, 0.8],
        [20, 30, 60, 70, 0.7]
    ])
    
    # 2个跟踪框 [x_min, y_min, x_max, y_max, age]
    previous_obs = np.array([
        [12, 22, 52, 62, 5],    # 有效跟踪
        [17, 27, 57, 67, -1]    # 无效跟踪 (age < 0)
    ])
    
    # 2个tracker框（用于计算scores形状）
    trackers = np.array([
        [12, 22, 52, 62],
        [17, 27, 57, 67]
    ])
    
    # 速度信息 [2个tracker, 5个点, 2个方向(dy, dx)]
    velocities = np.array([
        # tracker 0的5个点的速度
        [[0.1, 0.2], [0.15, 0.25], [0.12, 0.22], [0.14, 0.24], [0.13, 0.23]],
        # tracker 1的5个点的速度  
        [[0.2, 0.3], [0.25, 0.35], [0.22, 0.32], [0.24, 0.34], [0.23, 0.33]]
    ])
    
    vdc_weight = 0.5
    
    print("=== 原始方法 ===")
    # 原始方法调用
    Y_center, X_center = speed_direction_batch(detections[:, :4], previous_obs[:, :4])
    Y_lt, X_lt = speed_direction_batch_lt(detections[:, :4], previous_obs[:, :4])
    Y_rt, X_rt = speed_direction_batch_rt(detections[:, :4], previous_obs[:, :4])
    Y_lb, X_lb = speed_direction_batch_lb(detections[:, :4], previous_obs[:, :4])
    Y_rb, X_rb = speed_direction_batch_rb(detections[:, :4], previous_obs[:, :4])
    
    cost_center_orig = cost_vel(Y_center, X_center, trackers, velocities[:, 0], detections, previous_obs, vdc_weight)
    cost_lt_orig = cost_vel(Y_lt, X_lt, trackers, velocities[:, 1], detections, previous_obs, vdc_weight)
    cost_rt_orig = cost_vel(Y_rt, X_rt, trackers, velocities[:, 2], detections, previous_obs, vdc_weight)
    cost_lb_orig = cost_vel(Y_lb, X_lb, trackers, velocities[:, 3], detections, previous_obs, vdc_weight)
    cost_rb_orig = cost_vel(Y_rb, X_rb, trackers, velocities[:, 4], detections, previous_obs, vdc_weight)
    
    print("Center cost:")
    print(cost_center_orig)
    print("LT cost:")
    print(cost_lt_orig)
    print("RT cost:")
    print(cost_rt_orig)
    print("LB cost:")
    print(cost_lb_orig)
    print("RB cost:")
    print(cost_rb_orig)
    
    print("\n=== 新方法 ===")
    # 新方法调用
    directions = speed_direction_batch_all(detections[:, :4], previous_obs[:, :4])
    costs_new = cost_vel_batch(directions, trackers, velocities, detections, previous_obs, vdc_weight)
    
    cost_center_new, cost_lt_new, cost_rt_new, cost_lb_new, cost_rb_new = costs_new
    
    print("Center cost:")
    print(cost_center_new)
    print("LT cost:")
    print(cost_lt_new)
    print("RT cost:")
    print(cost_rt_new)
    print("LB cost:")
    print(cost_lb_new)
    print("RB cost:")
    print(cost_rb_new)
    
    print("\n=== 对比结果 ===")
    print(f"Center一致性: {np.allclose(cost_center_orig, cost_center_new, rtol=1e-10)}")
    print(f"LT一致性: {np.allclose(cost_lt_orig, cost_lt_new, rtol=1e-10)}")
    print(f"RT一致性: {np.allclose(cost_rt_orig, cost_rt_new, rtol=1e-10)}")
    print(f"LB一致性: {np.allclose(cost_lb_orig, cost_lb_new, rtol=1e-10)}")
    print(f"RB一致性: {np.allclose(cost_rb_orig, cost_rb_new, rtol=1e-10)}")
    
    # 详细差异检查
    print("\n=== 详细差异检查 ===")
    diffs = [
        ("Center", cost_center_orig, cost_center_new),
        ("LT", cost_lt_orig, cost_lt_new),
        ("RT", cost_rt_orig, cost_rt_new),
        ("LB", cost_lb_orig, cost_lb_new),
        ("RB", cost_rb_orig, cost_rb_new)
    ]
    
    for name, orig, new in diffs:
        max_diff = np.max(np.abs(orig - new))
        print(f"{name} 最大差异: {max_diff}")
        if max_diff > 1e-10:
            print(f"  原始: {orig}")
            print(f"  新的: {new}")
            print(f"  差异: {orig - new}")

if __name__ == "__main__":
    test_cost_vel()