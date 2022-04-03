import math
import random

import numpy as np


def _rotate(point, angle):
    """
    Rotate a point counterclockwise by a given angle around (0, 0).

    Angle is in radians.
    """
    px, py = point

    qx = math.cos(angle) * px - math.sin(angle) * py
    qy = math.sin(angle) * px + math.cos(angle) * py
    return qx, qy


def small_rotation(data, max_angle: float = 0.25, num_hands: int = 2):
    """
    Rotates image plane landmarks around (0, 0) which is hand's geometric center
    :param data: one set of 2d landmarks
    :param max_angle: maximum rotation in radians
    :param num_hands: the number of hands in the image
    :return: list
    """
    # Check if data is 2D (only if each hand gets 42 landmarks)
    if len(data) != num_hands * 42:
        raise ValueError('Data is not 2-dimensional')

    angle = random.uniform(-max_angle, max_angle)
    augmented = np.zeros(num_hands * 42)
    for i in range(0, num_hands * 42, 2):
        augmented[i], augmented[i+1] = _rotate((data[i], data[i+1]), angle)

    return augmented.tolist()


def scaling(data, max_factor: float = 1.2):
    """
    Multiplies all data points by a value.
    :param data: iterable (e.g. list of coordinates)
    :param max_factor: scaling is in range (1/max_factor, max_factor)
    :return: scaled data
    """
    factor = random.uniform(1 / max_factor, max_factor)
    for i in range(len(data)):
        data[i] = data[i] * factor
    return data
