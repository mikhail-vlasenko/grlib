import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse

from rotation_invariance.icp import point_correspondence_icp


def transform(landmarks_to_transform: np.ndarray, landmarks_anchor: np.ndarray, verbose=False):
    """
    Transforms a set of landmarks using rotation and translation to match the anchor as much as
    possible.
    :param landmarks_to_transform: array of landmarks that will be transformed
    :param landmarks_anchor: array of landmarks that will be used as the transformation anchor
    :param verbose: whether to print info
    :return: the transformed landmarks
    """
    scaler = MinMaxScaler()
    landmarks_to_transform = scaler.fit_transform(landmarks_to_transform.T)
    landmarks_anchor = scaler.fit_transform(landmarks_anchor.T)
    tr = point_correspondence_icp(landmarks_to_transform, landmarks_anchor, tolerance=0)[0]
    num = tr.shape[0] - 1
    r = tr[:num, :num]
    t = tr[:num, num]
    res = (landmarks_to_transform @ r.T + t)
    if verbose:
        start_mse = mse(landmarks_to_transform, landmarks_anchor)
        end_mse = mse(landmarks_anchor, res)
        relative_mse_change = (end_mse - start_mse) / start_mse
        print(f'from mse {start_mse} '
              f'to mse {end_mse} '
              f'(relative change: {relative_mse_change})')
    return res.T
