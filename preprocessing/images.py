import cv2.cv2 as cv
import imutils
import numpy as np


# Taken from https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
def increase_brightness(image: np.ndarray, value: float = 30) -> np.ndarray:
    """
    Update the brightness of the image by a given amount.
    :param image: the image to update
    :param value: by how much to change the brightness
    :return: updated image
    """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img


def rotate(image: np.ndarray, degrees: float = 0) -> np.ndarray:
    """
    Rotate the image by a given angle.
    :param image: the image to rotate
    :param degrees: the amount of degrees by which to rotate the image
    :return: rotated image
    """
    return imutils.rotate_bound(image, degrees)
