# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-arguments, too-many-branches,
# pylint: disable=too-many-locals, too-many-instance-attributes, too-many-lines

"""
This module implements the linear Kalman filter in both an object
oriented and procedural form. The KalmanFilter class implements
the filter by storing the various matrices in instance variables,
minimizing the amount of bookkeeping you have to do.
All Kalman filters operate with a predict->update cycle. The
predict step, implemented with the method or function predict(),
uses the state transition matrix F to predict the state in the next
time period (epoch). The state is stored as a gaussian (x, P), where
x is the state (column) vector, and P is its covariance. Covariance
matrix Q specifies the process covariance. In Bayesian terms, this
prediction is called the *prior*, which you can think of colloquially
as the estimate prior to incorporating the measurement.
The update step, implemented with the method or function `update()`,
incorporates the measurement z with covariance R, into the state
estimate (x, P). The class stores the system uncertainty in S,
the innovation (residual between prediction and measurement in
measurement space) in y, and the Kalman gain in k. The procedural
form returns these variables to you. In Bayesian terms this computes
the *posterior* - the estimate after the information from the
measurement is incorporated.
Whether you use the OO form or procedural form is up to you. If
matrices such as H, R, and F are changing each epoch, you'll probably
opt to use the procedural form. If they are unchanging, the OO
form is perhaps easier to use since you won't need to keep track
of these matrices. This is especially useful if you are implementing
banks of filters or comparing various KF designs for performance;
a trivial coding bug could lead to using the wrong sets of matrices.
This module also offers an implementation of the RTS smoother, and
other helper functions, such as log likelihood computations.
The Saver class allows you to easily save the state of the
KalmanFilter class after every update
This module expects NumPy arrays for all values that expect
arrays, although in a few cases, particularly method parameters,
it will accept types that convert to NumPy arrays, such as lists
of lists. These exceptions are documented in the method or function.
Examples
--------
The following example constructs a constant velocity kinematic
filter, filters noisy data, and plots the results. It also demonstrates
using the Saver class to save the state of the filter at each epoch.
.. code-block:: Python
    import matplotlib.pyplot as plt
    import numpy as np
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise, Saver
    r_std, q_std = 2., 0.003
    cv = KalmanFilter(dim_x=2, dim_z=1)
    cv.x = np.array([[0., 1.]]) # position, velocity
    cv.F = np.array([[1, dt],[ [0, 1]])
    cv.R = np.array([[r_std^^2]])
    f.H = np.array([[1., 0.]])
    f.P = np.diag([.1^^2, .03^^2)
    f.Q = Q_discrete_white_noise(2, dt, q_std**2)
    saver = Saver(cv)
    for z in range(100):
        cv.predict()
        cv.update([z + randn() * r_std])
        saver.save() # save the filter's state
    saver.to_array()
    plt.plot(saver.x[:, 0])
    # plot all of the priors
    plt.plot(saver.x_prior[:, 0])
    # plot mahalanobis distance
    plt.figure()
    plt.plot(saver.mahalanobis)
This code implements the same filter using the procedural form
    x = np.array([[0., 1.]]) # position, velocity
    F = np.array([[1, dt],[ [0, 1]])
    R = np.array([[r_std^^2]])
    H = np.array([[1., 0.]])
    P = np.diag([.1^^2, .03^^2)
    Q = Q_discrete_white_noise(2, dt, q_std**2)
    for z in range(100):
        x, P = predict(x, P, F=F, Q=Q)
        x, P = update(x, P, z=[z + randn() * r_std], R=R, H=H)
        xs.append(x[0, 0])
    plt.plot(xs)
For more examples see the test subdirectory, or refer to the
book cited below. In it I both teach Kalman filtering from basic
principles, and teach the use of this library in great detail.
FilterPy library.
http://github.com/rlabbe/filterpy
Documentation at:
https://filterpy.readthedocs.org
Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
This is licensed under an MIT license. See the readme.MD file
for more information.
Copyright 2014-2018 Roger R Labbe Jr.
"""

from __future__ import absolute_import, division

import pdb
from copy import deepcopy
from math import log, exp, sqrt
import sys
import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
import numpy.linalg as linalg
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z


class KalmanFilterNew(object):
    """Implements a Kalman filter. You are responsible for setting the
    various state variables to reasonable values; the defaults  will
    not give you a functional filter.
    For now the best documentation is my free book Kalman and Bayesian
    Filters in Python [2]_. The test files in this directory also give you a
    basic idea of use, albeit without much description.
    In brief, you will first construct this object, specifying the size of
    the state vector with dim_x and the size of the measurement vector that
    you will be using with dim_z. These are mostly used to perform size checks
    when you assign values to the various matrices. For example, if you
    specified dim_z=2 and then try to assign a 3x3 matrix to R (the
    measurement noise matrix you will get an assert exception because R
    should be 2x2. (If for whatever reason you need to alter the size of
    things midstream just use the underscore version of the matrices to
    assign directly: your_filter._R = a_3x3_matrix.)
    After construction the filter will have default matrices created for you,
    but you must specify the values for each. It’s usually easiest to just
    overwrite them rather than assign to each element yourself. This will be
    clearer in the example below. All are of type numpy.array.
    Examples
    --------
    Here is a filter that tracks position and velocity using a sensor that only
    reads position.
    First construct the object with the required dimensionality. Here the state
    (`dim_x`) has 2 coefficients (position and velocity), and the measurement
    (`dim_z`) has one. In FilterPy `x` is the state, `z` is the measurement.
    .. code::
        from filterpy.kalman import KalmanFilter
        f = KalmanFilter (dim_x=2, dim_z=1)
    Assign the initial value for the state (position and velocity). You can do this
    with a two dimensional array like so:
        .. code::
            f.x = np.array([[2.],    # position
                            [0.]])   # velocity
    or just use a one dimensional array, which I prefer doing.
    .. code::
        f.x = np.array([2., 0.])
    Define the state transition matrix:
        .. code::
            f.F = np.array([[1.,1.],
                            [0.,1.]])
    Define the measurement function. Here we need to convert a position-velocity
    vector into just a position vector, so we use:
        .. code::
        f.H = np.array([[1., 0.]])
    Define the state's covariance matrix P.
    .. code::
        f.P = np.array([[1000.,    0.],
                        [   0., 1000.] ])
    Now assign the measurement noise. Here the dimension is 1x1, so I can
    use a scalar
    .. code::
        f.R = 5
    I could have done this instead:
    .. code::
        f.R = np.array([[5.]])
    Note that this must be a 2 dimensional array.
    Finally, I will assign the process noise. Here I will take advantage of
    another FilterPy library function:
    .. code::
        from filterpy.common import Q_discrete_white_noise
        f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
    Now just perform the standard predict/update loop:
    .. code::
        while some_condition_is_true:
            z = get_sensor_reading()
            f.predict()
            f.update(z)
            do_something_with_estimate (f.x)
    **Procedural Form**
    This module also contains stand alone functions to perform Kalman filtering.
    Use these if you are not a fan of objects.
    **Example**
    .. code::
        while True:
            z, R = read_sensor()
            x, P = predict(x, P, F, Q)
            x, P = update(x, P, z, R, H)
    See my book Kalman and Bayesian Filters in Python [2]_.
    You will have to set the following attributes after constructing this
    object for the filter to perform properly. Please note that there are
    various checks in place to ensure that you have made everything the
    'correct' size. However, it is possible to provide incorrectly sized
    arrays such that the linear algebra can not perform an operation.
    It can also fail silently - you can end up with matrices of a size that
    allows the linear algebra to work, but are the wrong shape for the problem
    you are trying to solve.
    Parameters
    ----------
    dim_x : int
        Number of state variables for the Kalman filter. For example, if
        you are tracking the position and velocity of an object in two
        dimensions, dim_x would be 4.
        This is used to set the default size of P, Q, and u
    dim_z : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.
    dim_u : int (optional)
        size of the control input, if it is being used.
        Default value of 0 indicates it is not used.
    compute_log_likelihood : bool (default = True)
        Computes log likelihood by default, but this can be a slow
        computation, so if you never use it you can turn this computation
        off.
    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        Current state estimate. Any call to update() or predict() updates
        this variable.
    P : numpy.array(dim_x, dim_x)
        Current state covariance matrix. Any call to update() or predict()
        updates this variable.
    x_prior : numpy.array(dim_x, 1)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convenience; they store the  prior and posterior of the
        current epoch. Read Only.
    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.
    x_post : numpy.array(dim_x, 1)
        Posterior (updated) state estimate. Read Only.
    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.
    z : numpy.array
        Last measurement used in update(). Read only.
    R : numpy.array(dim_z, dim_z)
        Measurement noise covariance matrix. Also known as the
        observation covariance.
    Q : numpy.array(dim_x, dim_x)
        Process noise covariance matrix. Also known as the transition
        covariance.
    F : numpy.array()
        State Transition matrix. Also known as `A` in some formulation.
    H : numpy.array(dim_z, dim_x)
        Measurement function. Also known as the observation matrix, or as `C`.
    y : numpy.array
        Residual of the update step. Read only.
    K : numpy.array(dim_x, dim_z)
        Kalman gain of the update step. Read only.
    S :  numpy.array
        System uncertainty (P projected to measurement space). Read only.
    SI :  numpy.array
        Inverse system uncertainty. Read only.
    log_likelihood : float
        log-likelihood of the last measurement. Read only.
    likelihood : float
        likelihood of last measurement. Read only.
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
    mahalanobis : float
        mahalanobis distance of the innovation. Read only.
    inv : function, default numpy.linalg.inv
        If you prefer another inverse function, such as the Moore-Penrose
        pseudo inverse, set it to that instead: kf.inv = np.linalg.pinv
        This is only used to invert self.S. If you know it is diagonal, you
        might choose to set it to filterpy.common.inv_diagonal, which is
        several times faster than numpy.linalg.inv for diagonal matrices.
    alpha : float
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon [1]_.
    References
    ----------
    .. [1] Dan Simon. "Optimal State Estimation." John Wiley & Sons.
       p. 208-212. (2006)
    .. [2] Roger Labbe. "Kalman and Bayesian Filters in Python"
       https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError("dim_x must be 1 or greater")
        if dim_z < 1:
            raise ValueError("dim_z must be 1 or greater")
        if dim_u < 0:
            raise ValueError("dim_u must be 0 or greater")

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1))  # state
        self.P = eye(dim_x)  # uncertainty covariance
        self.Q = eye(dim_x)  # process uncertainty
        self.B = None  # control transition matrix
        self.F = eye(dim_x)  # state transition matrix
        self.H = zeros((dim_z, dim_x))  # measurement function
        # MOT17 [1, 1, 10, 0.1]
        # MOT20 [1, 1, 0.5, 5]
        # DANCE [1, 1, 0.5, 5]
        # self.R = eye(dim_z)  # measurement uncertainty        
        self.R = np.diag([1, 1, 100, 1])  # measurement uncertainty        
        self._alpha_sq = 1.0  # fading memory control
        self.M = np.zeros((dim_x, dim_z))  # process-measurement cross correlation
        self.z = np.array([[None] * self.dim_z]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((dim_x, dim_z))  # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))  # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # Only computed only if requested via property
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # keep all observations
        self.history_obs = []

        self.inv = np.linalg.inv

        self.attr_saved = None
        self.observed = False
        self.last_measurement = None
        
        self._std_weight_position = 1. / 20

    def predict(self, u=None, B=None, F=None, Q=None):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.
        Parameters
        ----------
        u : np.array, default 0
            Optional control vector.
        B : np.array(dim_x, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.
        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.
        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
        """

        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = dot(F, self.x) + dot(B, u)
        else:
            self.x = dot(F, self.x)

        # P = FPF' + Q
        self.P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def freeze(self):
        """
        Save the parameters before non-observation forward
        """
        self.attr_saved = deepcopy(self.__dict__)

    def apply_affine_correction(self, m, t):
        """
        Apply to both last state and last observation for OOS smoothing.

        Messy due to internal logic for kalman filter being messy.
        """
        scale = np.linalg.norm(m[:, 0])
        self.x[:2] = m @ self.x[:2] + t
        self.x[4:6] = m @ self.x[4:6]
        # self.x[2] *= scale
        # self.x[6] *= scale

        self.P[:2, :2] = m @ self.P[:2, :2] @ m.T
        self.P[4:6, 4:6] = m @ self.P[4:6, 4:6] @ m.T
        # self.P[2, 2] *= 2 * scale
        # self.P[6, 6] *= 2 * scale

        # If frozen, also need to update the frozen state for OOS
        if not self.observed and self.attr_saved is not None:
            self.attr_saved["x"][:2] = m @ self.attr_saved["x"][:2] + t
            self.attr_saved["x"][4:6] = m @ self.attr_saved["x"][4:6]
            # self.attr_saved["x"][2] *= scale
            # self.attr_saved["x"][6] *= scale

            self.attr_saved["P"][:2, :2] = m @ self.attr_saved["P"][:2, :2] @ m.T
            self.attr_saved["P"][4:6, 4:6] = m @ self.attr_saved["P"][4:6, 4:6] @ m.T
            # self.attr_saved["P"][2, 2] *= 2 * scale
            # self.attr_saved["P"][6, 6] *= 2 * scale

            self.attr_saved["last_measurement"][:2] = m @ self.attr_saved["last_measurement"][:2] + t
            # self.attr_saved["last_measurement"][2] *= scale

    def unfreeze(self):
        if self.attr_saved is not None:
            new_history = deepcopy(self.history_obs)
            self.__dict__ = self.attr_saved
            # self.history_obs = new_history
            self.history_obs = self.history_obs[:-1]
            occur = [int(d is None) for d in new_history]
            indices = np.where(np.array(occur) == 0)[0]
            index1 = indices[-2]
            index2 = indices[-1]
            # box1 = new_history[index1]
            box1 = self.last_measurement
            x1, y1, s1, r1 = box1
            w1 = np.sqrt(s1 * r1)
            h1 = np.sqrt(s1 / r1)

            box2 = new_history[index2]
            x2, y2, s2, r2 = box2
            w2 = np.sqrt(s2 * r2)
            h2 = np.sqrt(s2 / r2)

            time_gap = index2 - index1
            dx = (x2 - x1) / time_gap
            dy = (y2 - y1) / time_gap
            dw = (w2 - w1) / time_gap
            dh = (h2 - h1) / time_gap
            for i in range(index2 - index1):
                """
                The default virtual trajectory generation is by linear
                motion (constant speed hypothesis), you could modify this
                part to implement your own.
                """
                x = x1 + (i + 1) * dx
                y = y1 + (i + 1) * dy
                w = w1 + (i + 1) * dw
                h = h1 + (i + 1) * dh
                s = w * h
                r = w / float(h)
                new_box = np.array([x, y, s, r]).reshape((4, 1))
                """
                    I still use predict-update loop here to refresh the parameters,
                    but this can be faster by directly modifying the internal parameters
                    as suggested in the paper. I keep this naive but slow way for 
                    easy read and understanding
                """
                self.update(new_box)
                if not i == (index2 - index1 - 1):
                    self.predict()


    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter.
        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.
        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
            If you pass in a value of H, z must be a column vector the
            of the correct size.
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """
        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # append the observation
        self.history_obs.append(z)

        if z is None:
            if self.observed:
                """
                Got no observation so freeze the current parameters for future
                potential online smoothing.
                """
                self.last_measurement = self.history_obs[-2]
                self.freeze()
            self.observed = False
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        # self.observed = True
        if not self.observed:
            """
            Get observation, use online smoothing to re-update parameters
            """
            self.unfreeze()
        self.observed = True
        
        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R
            
        if H is None:
            z = reshape_z(z, self.dim_z, self.x.ndim)
            H = self.H
        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # common subexpression for speed
        PHT = dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        # print(R)
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
