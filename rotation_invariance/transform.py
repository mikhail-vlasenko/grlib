from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

from rotation_invariance.icp import point_correspondence_icp


def transform(landmarks_to_transform, landmarks_anchor):
    scaler = MinMaxScaler()
    landmarks_to_transform = scaler.fit_transform(landmarks_to_transform.T)
    landmarks_anchor = scaler.fit_transform(landmarks_anchor.T)
    tr = point_correspondence_icp(landmarks_to_transform, landmarks_anchor, tolerance=0)[0]
    num = tr.shape[0] - 1
    r = tr[:num, :num]
    t = tr[:num, num]
    res = (landmarks_to_transform @ r.T + t)
    print(f'from mse {mse(landmarks_to_transform, landmarks_anchor)} to mse {mse(landmarks_anchor, res)} '
          f'(share = {(mse(landmarks_anchor, res) - mse(landmarks_to_transform, landmarks_anchor)) / mse(landmarks_to_transform, landmarks_anchor)})')
    return res.T
