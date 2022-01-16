import math
import random

import numpy as np


def _rotate(point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    Angle is in radians.
    """
    px, py = point

    qx = math.cos(angle) * px - math.sin(angle) * py
    qy = math.sin(angle) * px + math.cos(angle) * py
    return qx, qy


def small_rotation(data, max_angle=0.25):
    """
    Rotates image plane landmarks around (0, 0) which is hand's geometric center
    :param data: one set of 2d landmarks (42 values)
    :param max_angle: maximum rotation in radians
    :return:
    """
    if len(data) != 42:
        raise ValueError
    angle = (random.random() - 0.5) * 2 * max_angle
    augmented = np.zeros(42)
    for i in range(0, 42, 2):
        augmented[i], augmented[i+1] = _rotate((data[i], data[i+1]), angle)

    return augmented.tolist()
