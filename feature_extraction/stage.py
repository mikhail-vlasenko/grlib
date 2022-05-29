import numpy as np

from preprocessing.images import increase_brightness, rotate


class Stage(object):

    def __init__(self, initial_index, brightness: float, rotation: float):
        self.initial_index = initial_index
        self.brightness = brightness
        self.rotation = rotation
        self.recognized_counter = 0

    def process(self, image: np.ndarray) -> np.ndarray:
        image = increase_brightness(image, self.brightness)
        image = rotate(image, self.rotation)

        return image
