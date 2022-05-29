from __future__ import annotations

from typing import Tuple, Any

import cv2.cv2 as cv
import numpy as np
from numpy import ndarray

from feature_extraction.mediapipe_landmarks import MediaPipe
from preprocessing.images import increase_brightness, rotate


class Stage(object):

    def __init__(self, mp: MediaPipe, initial_index: int, brightness: float, rotation: float):
        self.mp = mp
        self.initial_index = initial_index
        self.brightness = brightness
        self.rotation = rotation
        self.recognized_counter = 0
        self.last_detected_hands = None

    def process(self, image: np.ndarray) -> np.ndarray:
        image = increase_brightness(image, self.brightness)
        image = rotate(image, self.rotation)

        return image

    def run_get_landmarks(self, image: np.ndarray):
        converted_image = self.process(image)

        self.last_detected_hands = self.mp.process_from_image(converted_image).multi_hand_landmarks

    def run_get_world_landmarks(self, image: np.ndarray):
        converted_image = self.process(image)

        self.last_detected_hands = self.mp.process_from_image(converted_image).multi_hand_world_landmarks

