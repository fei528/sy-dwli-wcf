"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function
import math
import numpy as np
from .association import *
from .embedding import EmbeddingComputer
from .cmc import CMCComputer


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_bbox_to_z_new(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    return np.array([x, y, w, h]).reshape((4, 1))


def convert_x_to_bbox_new(x):
    x, y, w, h = x.reshape(-1)[:4]
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]).reshape(1, 4)


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm

# 过程噪声协方差矩阵
def new_kf_process_noise(w, h, p=1 / 20, v=1 / 160):
    Q = np.diag(
        (
            (p * w) ** 2,
            (p * h) ** 2,
            (p * w) ** 2,
            (p * h) ** 2,
            (v * w) ** 2,
            (v * h) ** 2,
            (v * w) ** 2,
            (v * h) ** 2,
        )
    )
    return Q


def new_kf_measurement_noise(w, h, m=1 / 20):
    w_var = (m * w) ** 2
    h_var = (m * h) ** 2
    R = np.diag((w_var, h_var, w_var, h_var))
    return R


def gaussian_amplified_kf_measurement_noise(w, h, m=1/20, confidence=0.0, alpha=100.0, threshold=0.7, sigma=0.3):
    """
    结合高斯函数和置信度阈值放大因子的测量噪声函数
    
    参数:
    w: float - 物体宽度
    h: float - 物体高度
    m: float - 基础噪声比例因子，默认为1/20
    confidence: float - 检测框置信度，范围[0,1]，默认为0.0
    alpha: float - 低置信度放大因子，默认为100.0
    threshold: float - 置信度阈值，默认为0.5
    sigma: float - 高斯函数的标准差参数，默认为0.3
    
    返回:
    R: ndarray - 测量噪声协方差矩阵
    """
    # 基础标准差计算
    base_std = [m * w, m * h, 1e-1, m * w]
    
    # 使用高斯函数计算基础噪声因子
    # 高斯函数在置信度=1处接近0，在置信度=0处接近1
    gaussian_factor = np.exp(-0.5 * (confidence**2) / (sigma**2))
    
    # 应用阈值放大机制
    scaling_factor = gaussian_factor
    if confidence < threshold:
        scaling_factor *= alpha
    
    # 应用缩放因子到标准差
    adjusted_std = [scaling_factor * x for x in base_std]
    
    # 计算协方差矩阵
    R = np.diag(np.square(adjusted_std))
    
    return R
    
def gaussian_amplified_kf_measurement_noise(w, h, m=1/20, confidence=0.0, alpha=100.0, threshold=0.7, sigma=0.3):
    """
    结合高斯函数和置信度阈值放大因子的测量噪声函数
    适用于状态向量: [x, y, s, r, x', y', s']
    测量向量: [x, y, s, r]
    
    参数:
    w: float - 物体宽度
    h: float - 物体高度
    m: float - 基础噪声比例因子，默认为1/20
    confidence: float - 检测框置信度，范围[0,1]，默认为0.0
    alpha: float - 低置信度放大因子，默认为100.0
    threshold: float - 置信度阈值，默认为0.7
    sigma: float - 高斯函数的标准差参数，默认为0.3
    
    返回:
    R: ndarray - 4x4测量噪声协方差矩阵，对应测量向量[x, y, s, r]
    """
    # 计算尺度 s = sqrt(w * h)
    s = np.sqrt(w * h)
    # 计算宽高比 r = w / h
    r = w / h if h > 0 else 1.0
    
    # 基础标准差计算，对应测量向量 [x, y, s, r]
    base_std = [
        m * w,      # x方向位置噪声，与宽度成比例
        m * h,      # y方向位置噪声，与高度成比例
        m * s,      # 尺度噪声，与尺度成比例
        0.1 * r     # 宽高比噪声，相对较小的固定比例
    ]
    
    # 使用高斯函数计算基础噪声因子
    # 高斯函数在置信度=1处接近0，在置信度=0处接近1
    gaussian_factor = np.exp(-0.5 * (confidence**2) / (sigma**2))
    
    # 应用阈值放大机制
    scaling_factor = gaussian_factor
    if confidence < threshold:
        scaling_factor *= alpha
    
    # 应用缩放因子到标准差
    adjusted_std = [scaling_factor * x for x in base_std]
    
    # 计算4x4协方差矩阵，对应测量向量 [x, y, s, r]
    R = np.diag(np.square(adjusted_std))
    
    return R

# 测量噪声协方差矩阵
def new_kf_measurement_noise_gs(w, h, m=1/20, confidence=0.0):
    """
    使用高斯函数计算测量噪声协方差矩阵
    
    参数:
    w: float - 宽度
    h: float - 高度
    m: float - 基础噪声比例因子，默认为1/20
    confidence: float - 检测框置信度，默认为0.0
    
    返回:
    R: ndarray - 测量噪声协方差矩阵
    """
    # 基础标准差计算
    std = [
        m * w,
        m * h,
        1e-1,
        m * w
    ]
    
    # 使用高斯函数根据置信度调整噪声，直接应用，不需要判断
    std = [((1 / (((2 * math.pi) ** 0.5))) * math.e ** (-(1-confidence) ** 2 / (2))) * x for x in std]
    
    # 计算协方差矩阵
    R = np.diag(np.square(std))
    return R


def new_kf_measurement_noise_nsa(w, h, m=1/20, confidence=0.0):
    """
    使用线性函数(R~k=(1−ck)Rk)计算测量噪声协方差矩阵
    
    参数:
    w: float - 宽度
    h: float - 高度
    m: float - 基础噪声比例因子，默认为1/20
    confidence: float - 检测框置信度，默认为0.0
    
    返回:
    R: ndarray - 测量噪声协方差矩阵
    """
    # 基础标准差计算
    std = [
        m * w,
        m * h,
        1e-1,
        m * w
    ]
    
    # 使用线性函数根据置信度调整噪声: R~k=(1−ck)Rk
    # 具体实现为: std~k=(1−ck)std_k
    std = [(1 - confidence) * x for x in std]
    
    # 计算协方差矩阵
    R = np.diag(np.square(std))
    return R


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, delta_t=3, orig=False, emb=None, alpha=0, r_thresh=20):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        if not orig:
            from .kalmanfilter import KalmanFilterNew as KalmanFilter
        else:
            from filterpy.kalman import KalmanFilter

        self.r_thresh = r_thresh
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                # x  y  s  r  x' y' s'
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.bbox_to_z_func = convert_bbox_to_z
        self.x_to_bbox_func = convert_x_to_bbox

        # Attempt
        # self.kf.P[2, 2] = 10000
        # self.kf.R[2, 2] = 10000

        self.kf.x[:4] = self.bbox_to_z_func(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        # Used for OCR
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        # Used to output track after min_hits reached
        self.history_observations = []
        # Used for velocity
        self.observations = dict()
        self.velocity = None
        self.delta_t = delta_t

        self.emb = emb

        self.frozen = False


    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            self.frozen = False

            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for dt in range(self.delta_t, 0, -1):
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(self.bbox_to_z_func(bbox), confidence=bbox[4], r_thresh=self.r_thresh)
        else:
            self.kf.update(bbox)
            self.frozen = True


    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb

    def apply_affine_correction(self, affine):
        m = affine[:, :2]
        t = affine[:, 2].reshape(2, 1)
        # For OCR
        if self.last_observation.sum() > 0:
            ps = self.last_observation[:4].reshape(2, 2).T
            ps = m @ ps + t
            self.last_observation[:4] = ps.T.reshape(-1)

        # Apply to each box in the range of velocity computation
        for dt in range(self.delta_t, -1, -1):
            if self.age - dt in self.observations:
                ps = self.observations[self.age - dt][:4].reshape(2, 2).T
                ps = m @ ps + t
                self.observations[self.age - dt][:4] = ps.T.reshape(-1)

        # Also need to change kf state, but might be frozen
        self.kf.apply_affine_correction(m, t)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # Don't allow negative bounding boxes
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        Q = None

        self.kf.predict(Q=Q)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.x_to_bbox_func(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def mahalanobis(self, bbox):
        """Should be run after a predict() call for accuracy."""
        return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist,
}


class BeyondTrack(object):
    def __init__(
        self,
        det_thresh,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        w_association_emb=0.75,
        alpha_fixed_emb=0.95,
        aw_param=0.5,
        embedding_off=False,
        use_cmc=False,
        aw_off=False,
        grid_off=False,
        r_thresh=20,
        **kwargs,
    ):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        KalmanBoxTracker.count = 0

        self.embedder = EmbeddingComputer(kwargs["args"].dataset, kwargs["args"].test_dataset, grid_off)
        self.cmc = CMCComputer()
        self.embedding_off = embedding_off
        self.use_cmc = use_cmc
        self.aw_off = aw_off
        self.grid_off = grid_off
        self.r_thresh = r_thresh


    def update(self, output_results, img_tensor, img_numpy, tag):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))
        if not isinstance(output_results, np.ndarray):
            output_results = output_results.cpu().numpy()
        self.frame_count += 1
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        # Rescale
        scale = min(img_tensor.shape[2] / img_numpy.shape[0], img_tensor.shape[3] / img_numpy.shape[1])
        dets[:, :4] /= scale

        # Generate embeddings
        dets_embs = np.ones((dets.shape[0], 1))
        if not self.embedding_off and dets.shape[0] != 0:
            # Shape = (num detections, 3, 512) if grid
            dets_embs = self.embedder.compute_embedding(img_numpy, dets[:, :4], tag)


        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = self.alpha_fixed_emb
        # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
        dets_alpha = af + (1 - af) * (1 - trust)

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        trk_embs = []
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.trackers[t].get_emb())
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # Shape = (num_trackers, 3, 512) if grid
        trk_embs = np.array(trk_embs)
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        """
            First round of association
        """
        matched, unmatched_dets, unmatched_trks = associate(
            dets,
            trks,
            dets_embs,
            trk_embs,
            self.iou_threshold,
            velocities,
            k_observations,
            self.inertia,
            self.w_association_emb,
            self.aw_off,
            self.aw_param,
            self.embedding_off,
            self.grid_off,
        )
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])
        """
            Second round of associaton by OCR
        """
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_dets_embs = dets_embs[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_trks_embs = trk_embs[unmatched_trks]

            # TODO: maybe use embeddings here
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)

                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    self.trackers[trk_ind].update_emb(dets_embs[det_ind], alpha=dets_alpha[det_ind])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i, :], delta_t=self.delta_t, emb=dets_embs[i], alpha=dets_alpha[i],  r_thresh=self.r_thresh
            )
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                this is optional to use the recent observation or the kalman filter prediction,
                we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def dump_cache(self):
        # self.cmc.dump_cache()
        self.embedder.dump_cache()
