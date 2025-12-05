# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 3/11/2019

from __future__ import division

from copy import deepcopy
from typing import Optional, Dict

import numpy as np
import cv2
import os
import json


def ecc(src, dst, warp_mode=cv2.MOTION_EUCLIDEAN, eps=1e-5, max_iter=100, scale=0.1, align=False):
    """Compute the warp matrix from src to dst.

    Parameters
    ----------
    src : ndarray
        An NxM matrix of source img(BGR or Gray), it must be the same format as dst.
    dst : ndarray
        An NxM matrix of target img(BGR or Gray).
    warp_mode: flags of opencv
        translation: cv2.MOTION_TRANSLATION
        rotated and shifted: cv2.MOTION_EUCLIDEAN
        affine(shift,rotated,shear): cv2.MOTION_AFFINE
        homography(3d): cv2.MOTION_HOMOGRAPHY
    eps: float
        the threshold of the increment in the correlation coefficient between two iterations
    max_iter: int
        the number of iterations.
    scale: float or [int, int]
        scale_ratio: float
        scale_size: [W, H]
    align: bool
        whether to warp affine or perspective transforms to the source image

    Returns
    -------
    warp matrix : ndarray
        Returns the warp matrix from src to dst.
        if motion models is homography, the warp matrix will be 3x3, otherwise 2x3
    src_aligned: ndarray
        aligned source image of gray
    """
    assert src.shape == dst.shape, "the source image must be the same format to the target image!"

    # BGR2GRAY
    if src.ndim == 3:
        # Convert images to grayscale
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # make the imgs smaller to speed up
    if scale is not None:
        if isinstance(scale, float):
            if scale != 1:
                src_r = cv2.resize(src, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                dst_r = cv2.resize(dst, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                scale = [scale, scale]
            else:
                src_r, dst_r = src, dst
                scale = None
        elif isinstance(scale, int):
            scale = scale / src.shape[1]
            src_r = cv2.resize(src, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            dst_r = cv2.resize(dst, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            scale = [scale, scale]
        else:
            if scale[0] != src.shape[1] and scale[1] != src.shape[0]:
                src_r = cv2.resize(src, (scale[0], scale[1]), interpolation=cv2.INTER_LINEAR)
                dst_r = cv2.resize(dst, (scale[0], scale[1]), interpolation=cv2.INTER_LINEAR)
                scale = [scale[0] / src.shape[1], scale[1] / src.shape[0]]
            else:
                src_r, dst_r = src, dst
                scale = None
    else:
        src_r, dst_r = src, dst

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(src_r, dst_r, warp_matrix, warp_mode, criteria, None, 1)

    if scale is not None:
        warp_matrix[0, 2] = warp_matrix[0, 2] / scale[0]
        warp_matrix[1, 2] = warp_matrix[1, 2] / scale[1]

    if align:
        sz = src.shape
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            src_aligned = cv2.warpPerspective(src, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            src_aligned = cv2.warpAffine(src, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR)
        return warp_matrix, src_aligned
    else:
        return warp_matrix, None


def Sparse_Flow(prev_image, frame, prev_desc, mask):
    # Initialize
    A = np.eye(2, 3)
    sparse_flow_param = dict(
        maxCorners=3000,
        qualityLevel=0.01,
        minDistance=1,
        blockSize=3,
        useHarrisDetector=False,
        k=0.04,
    )
    # find the keypoints
    keypoints = cv2.goodFeaturesToTrack(frame, mask=mask, **sparse_flow_param)
    prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)

    matched_kp, status, err = cv2.calcOpticalFlowPyrLK(prev_image, frame, prev_desc, None)
    matched_kp = matched_kp.reshape(-1, 2)
    status = status.reshape(-1)
    prev_points = prev_desc.reshape(-1, 2)
    prev_points = prev_points[status]
    curr_points = matched_kp[status]

    # Find rigid matrix
    if prev_points.shape[0] > 10:
        A, _ = cv2.estimateAffinePartial2D(prev_points, curr_points, method=cv2.RANSAC)
    else:
        print("Warning: not enough matching points")
    if A is None:
        A = np.eye(2, 3)

    # prev_image = frame
    # prev_desc = keypoints
    return A, keypoints


class ECC:

    def __init__(
        self,
        warp_mode=cv2.MOTION_EUCLIDEAN,
        eps=1e-4,
        max_iter=100,
        scale=0.15,
        align=False,
        video_name: Optional[str] = None,
        use_cache: bool = True,
    ):
        self.wrap_mode = warp_mode
        self.eps = eps
        self.max_iter = max_iter
        self.scale = scale
        self.align = align
        self.prev_image: Optional[np.ndarray] = None
        self.prev_desc: Optional[np.ndarray] = None
        self.video_name = video_name
        self.use_cache = use_cache
        self.cache: Dict[str, np.ndarray] = dict()
        if self.use_cache and self.video_name is not None and len(self.video_name) > 0:
            try:
                self.cache = json.load(open(os.path.join("cache", "ecc", self.video_name + ".json"), "r"))
                for k in self.cache:
                    self.cache[k] = np.array(self.cache[k])
                if len(self.cache) > 1:
                    print("USING CMC CACHE!")
            except:
                pass

    def __call__(self, np_image: np.ndarray, frame_id: int, video: Optional[str] = "", mode="sparse") -> np.ndarray:
        if frame_id == 1:
            self.prev_image = deepcopy(np_image)
            img = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
            sparse_flow_param = dict(
                maxCorners=3000,
                qualityLevel=0.01,
                minDistance=1,
                blockSize=3,
                useHarrisDetector=False,
                k=0.04,
            )
            mask = np.ones_like(img, dtype=np.uint8)
            # find the keypoints
            self.prev_desc = cv2.goodFeaturesToTrack(img, mask=mask, **sparse_flow_param)
            return np.eye(3, dtype=float)
        key = "{}-{}".format(video, frame_id)
        if key in self.cache:
            return self.cache[key]
        if mode == "ecc":
            result, _ = ecc(self.prev_image, np_image, self.wrap_mode, self.eps, self.max_iter, self.scale, self.align)
        else:
            img = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
            mask = np.ones_like(img, dtype=np.uint8)
            result, self.prev_desc = Sparse_Flow(self.prev_image, img, self.prev_desc, mask)
        self.prev_image = deepcopy(np_image)

        if self.use_cache:
            self.cache[key] = deepcopy(result)

        return result

    def save_cache(self):
        if not self.use_cache:
            return
        if self.video_name is not None and len(self.video_name) > 0:
            os.makedirs(os.path.join("cache", "ecc"), exist_ok=True)
            f = open(os.path.join("cache", "ecc", self.video_name + ".json"), "w")
            for k in self.cache:
                self.cache[k] = self.cache[k].tolist()
            json.dump(self.cache, f)
            f.close()
