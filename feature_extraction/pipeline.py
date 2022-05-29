from multiprocessing.pool import ThreadPool
from typing import List

import numpy as np
from exceptions import NoHandDetectedException
from feature_extraction.mediapipe_landmarks import MediaPipe
from feature_extraction.stage import Stage
import cv2.cv2 as cv


def run_stage_landmarks(zipped):
    stage, image = zipped
    stage.run_get_landmarks(image)


def run_stage_world_landmarks(zipped):
    stage, image = zipped
    stage.run_get_world_landmarks(image)


class Pipeline(object):

    def __init__(self, num_hands: int = 2, optimize_pipeline: bool = True):
        self.num_hands = num_hands
        self.optimize_pipeline = optimize_pipeline
        self.total = 0
        self.stages: List[Stage] = []
        self.thread_pool = ThreadPool(1)

    def add_stage(self, brightness: float = 0, rotation: float = 0):
        new_mp = MediaPipe(self.num_hands)
        stage = Stage(new_mp, len(self.stages), brightness, rotation)
        self.stages.append(stage)
        self.thread_pool = ThreadPool(len(self.stages))

    def optimize(self):
        if self.optimize_pipeline:
            self.stages = sorted(self.stages, key=lambda stage: stage.recognized_counter, reverse=True)

    def run_pipeline(self, img_path: str, callback) -> np.ndarray:
        image = cv.imread(img_path)

        # Reset last_detected_hands for every stage
        for stage in self.stages:
            stage.last_detected_hands = None

        self.thread_pool.map(callback, zip(self.stages, [image for _ in range(len(self.stages))]))

        # Find detections (if any)
        self.total += 1
        for stage in self.stages:
            detected_hands = stage.last_detected_hands
            if detected_hands is not None:
                stage.recognized_counter += 1
                return stage.mp.get_landmarks_from_hands(detected_hands)

        raise NoHandDetectedException(f'No hand detected for {img_path}')

    def get_landmarks(self, img_path: str) -> np.ndarray:
        return self.run_pipeline(img_path, run_stage_landmarks)

    def get_world_landmarks(self, img_path: str) -> np.ndarray:
        return self.run_pipeline(img_path, run_stage_world_landmarks)

    def __str__(self) -> str:
        total_recognized = sum(stage.recognized_counter for stage in self.stages)
        recognition_rate = round(total_recognized / self.total * 100, 2)
        order = ' -> '.join(f'{stage.initial_index} [{stage.recognized_counter}]' for stage in self.stages)
        order += f' -> fail [{self.total - total_recognized}]'

        return f'Recognized {total_recognized}/{self.total} [{recognition_rate}%]: pipeline = {order}'
