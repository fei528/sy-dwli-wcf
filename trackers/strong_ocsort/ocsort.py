"""
This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""

from __future__ import print_function

import numpy as np
from .association import *
from .embedding import EmbeddingComputer
from .cmc import CMCComputer
from .ecc import ECC
import cv2
from copy import deepcopy


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
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    score = bbox[4]
    return np.array([x, y, w, h, score]).reshape((5, 1))


def convert_x_to_bbox(x):
    x, y, w, h, score = x.reshape(-1)[:5]
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2, score]).reshape(1, 5)


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_all(bbox1, bbox2):
    """
    计算边界框中心点和四个角点的速度方向

    Args:
        bbox1, bbox2: 边界框 [x_min, y_min, x_max, y_max]

    Returns:
        tuple: (center, lt, rt, lb, rb)
               每个都是归一化的速度方向向量 [dy, dx]
    """
    # 定义所有关键点的坐标 (x1, y1, x2, y2)
    points = [
        (
            (bbox1[0] + bbox1[2]) / 2.0,
            (bbox1[1] + bbox1[3]) / 2.0,
            (bbox2[0] + bbox2[2]) / 2.0,
            (bbox2[1] + bbox2[3]) / 2.0,
        ),  # center
        (bbox1[0], bbox1[1], bbox2[0], bbox2[1]),  # lt
        (bbox1[2], bbox1[1], bbox2[2], bbox2[1]),  # rt
        (bbox1[0], bbox1[3], bbox2[0], bbox2[3]),  # lb
        (bbox1[2], bbox1[3], bbox2[2], bbox2[3]),  # rb
    ]

    # 计算每个点的速度方向
    results = []
    for x1, y1, x2, y2 in points:
        dy, dx = y2 - y1, x2 - x1
        norm = np.sqrt(dy * dy + dx * dx) + 1e-6
        results.append(np.array([dy, dx]) / norm)

    return results


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, delta_t=3, emb=None, track_thresh=None):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        from .kalmanfilter import KalmanFilterNew as KalmanFilter

        self.alpha = 1.0
        self.const_noise = False
        _, _, w, h, _ = convert_bbox_to_z(bbox).reshape(-1)

        if w * h > 15000:
            self.alpha = w * h / 5000
            self.P = self.state_noise(w, h)
            self.kf = KalmanFilter(dim_x=10, dim_z=5, P=self.P)
            self.kf.R = self.measurement_noise(w, h)
            self.kf.Q = self.process_noise(w, h)
        else:
            self.kf = KalmanFilter(dim_x=10, dim_z=5)
            self.kf.P[4, 4] *= 10
            # give high uncertainty to the unobservable initial velocities
            self.kf.P[5:, 5:] *= 1000.0
            self.kf.P[-1, -1] *= 10
            self.kf.Q[4:, 4:] *= 0.01
            self.kf.Q[-1, -1] *= 0.01
            self.kf.R = np.diag([10, 30, 100, 500, 10])
            self.const_noise = True

        self.kf.F = np.array(
            [
                # x y  w  h  c  x' y' w' h' c'
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ]
        )
        # Process and measurement uncertainty happen in functions
        self.bbox_to_z_func = convert_bbox_to_z
        self.x_to_bbox_func = convert_x_to_bbox

        self.kf.x[:5] = self.bbox_to_z_func(bbox)

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
        self.velocities_list = None
        self.delta_t = delta_t
        self.emb = emb
        self.frozen = False

        self.confidence_pre = None
        self.confidence = bbox[-1]
        self.track_thresh = track_thresh

    """
    过程噪声
    """

    def process_noise(self, w, h, p=1 / 20, v=1 / 160):
        Q = [
            p * w,
            p * h,
            p * w,
            p * h,
            1,
            v * w,
            v * h,
            v * w * 0.01,
            v * h * 0.01,
            0.01,
        ]
        return np.diag(np.square(Q))

    """
    状态噪声
    """

    def state_noise(self, w, h, p=1 / 20, v=1 / 160):
        values = [
            2 * p * w,
            2 * p * h,
            2 * p * w,
            2 * p * h,
            10,
            10 * v * w,
            10 * v * h,
            10 * v * w * 0.01,
            10 * v * h * 0.01,
            100,
        ]
        return np.diag(np.square(values))

    """
    观测噪声
    """

    def measurement_noise(self, w, h, m=1 / 20, n=1 / 5):
        # std = [
        #         m * w,
        #         m * h,
        #         n * w * self.alpha,
        #         n * h * self.alpha,
        # ]
        # R = np.diag(
        #     np.square(std)

        # x,y,w,h,c
        R = np.diag([10, 30, 100 * self.alpha, 500 * self.alpha, 10])
        return R

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            self.frozen = False

            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    # dt = self.delta_t - i
                    if self.age - i - 1 in self.observations:
                        previous_box = self.observations[self.age - i - 1]
                if previous_box is None:
                    # 使用last_observation作为参考
                    previous_box = self.last_observation
                self.velocities_list = speed_direction_all(previous_box, bbox)

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
            self.confidence_pre = self.confidence
            self.confidence = bbox[-1]

            if self.const_noise:
                self.kf.update(self.bbox_to_z_func(bbox))
            else:
                R = self.measurement_noise(self.kf.x[2, 0], self.kf.x[3, 0])
                self.kf.update(self.bbox_to_z_func(bbox), R=R)
        else:
            self.kf.update(bbox)
            self.frozen = True

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb

    def get_confidence(self, coef: float = 0.9) -> float:
        n = 7

        if self.age < n:
            return coef ** (n - self.age)
        return coef ** (self.time_since_update - 1)

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
        if self.kf.x[2] + self.kf.x[7] <= 0:
            self.kf.x[7] = 0
        if self.kf.x[3] + self.kf.x[8] <= 0:
            self.kf.x[8] = 0

        # Stop velocity, will update in kf during OOS
        if self.frozen:
            self.kf.x[7] = self.kf.x[8] = 0

        if self.const_noise:
            Q = None
        else:
            Q = self.process_noise(self.kf.x[2, 0], self.kf.x[3, 0])

        self.kf.predict(Q=Q)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.x_to_bbox_func(self.kf.x))
        if not self.confidence_pre:
            return (
                self.history[-1],
                np.clip(self.kf.x[4], self.track_thresh, 1.0),
            )
        else:
            return (
                self.history[-1],
                np.clip(self.kf.x[4], self.track_thresh, 1.0),
            )

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
    "hmiou": hmiou,
}


class OCSort(object):
    def __init__(
        self,
        det_thresh,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.05,
        w_association_emb=0.75,
        alpha_fixed_emb=0.95,
        aw_param=0.5,
        embedding_off=False,
        cmc_off=False,
        aw_off=False,
        tcm_weight=1.0,
        video_name=None,
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

        self.embedder = EmbeddingComputer(kwargs["args"].dataset, kwargs["args"].test_dataset)
        self.cmc = CMCComputer()
        self.ecc = ECC(scale=350, video_name=video_name, use_cache=True)

        self.embedding_off = embedding_off
        self.cmc_off = cmc_off
        self.aw_off = aw_off
        # 调试模式，打印图片
        self.visual = kwargs["args"].visual
        self.track_thresh = kwargs["args"].track_thresh
        # 启用TCM
        self.tcm_weight = tcm_weight

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
        scale = min(
            img_tensor.shape[2] / img_numpy.shape[0],
            img_tensor.shape[3] / img_numpy.shape[1],
        )
        dets[:, :4] /= scale

        # Generate embeddings
        dets_embs = np.ones((dets.shape[0], 1))
        if not self.embedding_off and dets.shape[0] != 0:
            # Shape = (num detections, 3, 512) if grid
            dets_embs = self.embedder.compute_embedding(img_numpy, dets[:, :4], tag)

        # CMC
        if not self.cmc_off:
            transform = self.cmc.compute_affine(img_numpy, dets[:, :4], tag)
            # transform = self.ecc(img_numpy, self.frame_count, tag)
            for trk in self.trackers:
                trk.apply_affine_correction(transform)

        # dets = self.dlo_confidence_boost(dets, True, True, True)

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
            pos, kalman_score = self.trackers[t].predict()
            try:
                trk[:] = [
                    pos[0][0],
                    pos[0][1],
                    pos[0][2],
                    pos[0][3],
                    kalman_score,
                ]
            except:
                pos = np.squeeze(pos)
                trk[:] = [
                    pos[0],
                    pos[1],
                    pos[2],
                    pos[3],
                    float(kalman_score),
                ]

            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.trackers[t].get_emb())
                # if self.visual:
                #     # 画出预测框
                #     x1, y1, x2, y2 = map(int, pos[:4])
                #     cv2.rectangle(img_numpy, (x1, y1), (x2, y2), (255, 255, 255), 2)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # Shape = (num_trackers, 3, 512) if grid
        trk_embs = np.array(trk_embs)
        for t in reversed(to_del):
            self.trackers.pop(t)

        # velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        velocities = np.array(
            [
                (trk.velocities_list if trk.velocities_list is not None else [np.array((0, 0)) for _ in range(5)])
                for trk in self.trackers
            ]
        )
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])
        """
            First round of association
            一阶段匹配：将高置信度检测框与未匹配的检测框进行匹配
        """
        matched, unmatched_dets, unmatched_trks = associate_with_corner(
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
            self.asso_func,
            self.tcm_weight,
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
                # if iou_left.max() > 0.1:
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
                dets[i, :],
                delta_t=self.delta_t,
                emb=dets_embs[i],
                track_thresh=self.track_thresh,
            )
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0][:4]
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

    def get_mh_dist_matrix(self, detections: np.ndarray, n_dims: int = 4) -> np.ndarray:
        if len(self.trackers) == 0:
            return np.zeros((0, 0))
        z = np.zeros((len(detections), n_dims), dtype=float)
        x = np.zeros((len(self.trackers), n_dims), dtype=float)
        sigma_inv = np.zeros_like(x, dtype=float)

        f = self.trackers[0].bbox_to_z_func
        for i in range(len(detections)):
            z[i, :n_dims] = f(detections[i, :]).reshape((-1,))[:n_dims]
        for i in range(len(self.trackers)):
            # 修改这里：确保展平为一维数组
            x[i] = self.trackers[i].kf.x[:n_dims].reshape(-1)  # 或者用 .flatten()
            # Note: we assume diagonal covariance matrix
            sigma_inv[i] = np.reciprocal(np.diag(self.trackers[i].kf.P[:n_dims, :n_dims]))

        return (
            (z.reshape((-1, 1, n_dims)) - x.reshape((1, -1, n_dims))) ** 2 * sigma_inv.reshape((1, -1, n_dims))
        ).sum(axis=2)

    def get_iou_matrix(self, detections: np.ndarray) -> np.ndarray:
        trackers = np.zeros((len(self.trackers), 5))
        for t, trk in enumerate(trackers):
            pos = self.trackers[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], self.trackers[t].get_confidence()]

        return self.asso_func(detections, trackers)

    def dlo_confidence_boost(
        self, detections: np.ndarray, use_rich_sim: bool, use_soft_boost: bool, use_varying_th: bool
    ) -> np.ndarray:
        sbiou_matrix = self.get_iou_matrix(detections)
        if sbiou_matrix.size == 0:
            return detections
        trackers = np.zeros((len(self.trackers), 6))
        for t, trk in enumerate(trackers):
            pos = self.trackers[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, self.trackers[t].time_since_update - 1]
        if use_rich_sim:
            mhd_sim = MhDist_similarity(self.get_mh_dist_matrix(detections), 1)
            S = (mhd_sim + sbiou_matrix) / 2
        else:
            S = self.get_iou_matrix(detections, False)

        if not use_soft_boost and not use_varying_th:
            """
            "mot17": {"dlo_boost_coef": 0.65},
            "mot20": {"dlo_boost_coef": 0.5},
            "dance": {"dlo_boost_coef": 0.5},
            """
            max_s = S.max(1)
            coef = 0.65
            detections[:, 4] = np.maximum(detections[:, 4], max_s * coef)

        else:
            if use_soft_boost:
                max_s = S.max(1)
                alpha = 0.65
                detections[:, 4] = np.maximum(detections[:, 4], alpha * detections[:, 4] + (1 - alpha) * max_s ** (1.5))
            if use_varying_th:
                threshold_s = 0.95
                threshold_e = 0.8
                n_steps = 3
                alpha = (threshold_s - threshold_e) / n_steps
                tmp = (S > np.maximum(threshold_s - trackers[:, 5] * alpha, threshold_e)).max(1)
                scores = deepcopy(detections[:, 4])
                scores[tmp] = np.maximum(scores[tmp], self.det_thresh + 1e-5)

                detections[:, 4] = scores

        return detections
