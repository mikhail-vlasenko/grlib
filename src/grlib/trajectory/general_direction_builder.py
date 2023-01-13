from enum import Enum
from typing import List, Union

import numpy as np

from src.grlib.feature_extraction.mediapipe_landmarks import MediaPipe


class Direction(Enum):
    UP = 1
    RIGHT = UP
    INTO = UP
    STATIONARY = 0
    DOWN = -1
    LEFT = DOWN
    AWAY = DOWN


class GeneralDirectionBuilder:
    """
    Composes a trajectory as a sequence of Direction enum(1/0/-1) on multiple axi.
    Inspired by https://ieeexplore-ieee-org.tudelft.idm.oclc.org/stamp/stamp.jsp?tp=&arnumber=485888
    """
    def __init__(self, zero_precision: float = 0.1):
        """

        :param zero_precision: how much is considered "no movement on the axis"
        """
        self.dimensions = 3  # x, y, and z (always 3 for consistency)
        self.zero_precision = zero_precision

    def make_trajectory(
            self,
            landmark_sequence: Union[np.ndarray, List[np.ndarray]],
            hand_num: int = 0
    ) -> np.ndarray:
        """
        Creates the trajectory
        :param landmark_sequence: a sequence of landmarks from (sample) frames
        :param hand_num: index of hand to make the trajectory for
        :return: the trajectory encoding as a 2d numpy array.
        3 columns (for x, y, z) and len(landmark_sequence) rows
        """
        trajectory = []
        last = MediaPipe.hands_spacial_position(landmark_sequence[0])[hand_num]
        for landmark in landmark_sequence[1:]:
            directions = []
            current = MediaPipe.hands_spacial_position(landmark)[hand_num]
            for i in range(self.dimensions):
                lower_boundary = last[i] - self.zero_precision
                upper_boundary = last[i] + self.zero_precision
                if lower_boundary > current[i]:
                    directions.append(Direction.DOWN.value)
                elif upper_boundary < current[i]:
                    directions.append(Direction.UP.value)
                else:
                    directions.append(Direction.STATIONARY.value)
            trajectory.append(directions)
            last = current
        return np.array(trajectory)
