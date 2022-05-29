from typing import List

import numpy as np

from feature_extraction.stage import Stage


class Pipeline(object):

    def __init__(self, optimize_pipeline: bool = True):
        self.optimize_pipeline = optimize_pipeline
        self.total = 0
        self.stages: List[Stage] = []

    def add_stage(self, brightness: float = 0, rotation: float = 0):
        stage = Stage(len(self.stages), brightness, rotation)
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
