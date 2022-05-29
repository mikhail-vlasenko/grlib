from typing import List

import numpy as np

import cv2.cv2 as cv
from exceptions import NoHandDetectedException
from feature_extraction.mediapipe_landmarks import MediaPipe
from feature_extraction.stage import Stage


class Pipeline(object):

    def __init__(self, num_hands: int = 2, optimize_pipeline: bool = True):
        self.num_hands = num_hands
        self.optimize_pipeline = optimize_pipeline
        self.total = 0
        self.stages: List[Stage] = []

    def add_stage(self, brightness: float = 0, rotation: float = 0):
        new_mp = MediaPipe(self.num_hands)
        stage = Stage(new_mp, len(self.stages), brightness, rotation)
        self.stages.append(stage)

    def __str__(self) -> str:
        total_recognized = sum(stage.recognized_counter for stage in self.stages)
        recognition_rate = round(total_recognized / self.total * 100, 2)
        order = ' -> '.join(f'{stage.initial_index} [{stage.recognized_counter}]' for stage in self.stages)
        order += f' -> fail [{self.total - total_recognized}]'

        return f'Recognized {total_recognized}/{self.total} [{recognition_rate}%]: pipeline = {order}'

    def optimize(self):
        if self.optimize_pipeline:
            self.stages = sorted(self.stages, key=lambda stage: stage.recognized_counter, reverse=True)

    def get_landmarks(self, img_path: str) -> np.ndarray:
        image = cv.imread(img_path)

        self.total += 1
        for stage in self.stages:
            detected_hands = stage.get_landmarks(image)
            if detected_hands is not None:
                stage.recognized_counter += 1
                return stage.mp.get_landmarks_from_hands(detected_hands)

        raise NoHandDetectedException(f'No hand detected for {img_path}')

    def get_world_landmarks(self, img_path: str) -> np.ndarray:
        image = cv.imread(img_path)

        self.total += 1
        for stage in self.stages:
            detected_hands = stage.get_world_landmarks(image)
            if detected_hands is not None:
                stage.recognized_counter += 1
                return stage.mp.get_landmarks_from_hands(detected_hands)

        raise NoHandDetectedException(f'No hand detected for {img_path}')

