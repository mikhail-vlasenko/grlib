import numpy as np

from src.feature_extraction.mediapipe_landmarks import MediaPipe
from src.preprocessing.images import increase_brightness, rotate


class Stage(object):
    """
    Class to represent a single stage of the pipeline.
    """

    def __init__(self, mp: MediaPipe, initial_index: int, brightness: float, rotation: float):
        """
        :param mp: mediapipe object to be used for processing
        :param initial_index: the index of the stage in the pipeline before any order optimizations happened
        :param brightness: the brightness that will be added to each image
        :param rotation: the rotation that will be applied to each image
        """
        self.mp = mp
        self.initial_index = initial_index
        self.brightness = brightness
        self.rotation = rotation
        self.recognized_counter = 0
        self.last_detected_hands = None

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process the image. Apply brightness and rotation.
        :param image: the image to process
        :return: processed image
        """
        image = increase_brightness(image, self.brightness)
        image = rotate(image, self.rotation)

        return image

    def get_landmarks(self, image: np.ndarray):
        """
        Gets the mediapipe landmarks from an image. Saves them to self.last_detected_hands because
        the method is called asynchronously (so can't really return anything).
        :param image: the image to process
        """
        converted_image = self.process(image)

        self.last_detected_hands = self.mp.process_from_image(converted_image).multi_hand_landmarks

    def get_world_landmarks(self, image: np.ndarray):
        """
        Gets the mediapipe world landmarks from an image. Saves them to self.last_detected_hands because
        the method is called asynchronously (so can't really return anything).
        :param image: the image to process
        """
        converted_image = self.process(image)

        self.last_detected_hands = self.mp.process_from_image(converted_image).multi_hand_world_landmarks

