from enum import Enum
from typing import List, Union

import numpy as np

from src.grlib.feature_extraction.mediapipe_landmarks import MediaPipe


DIMENSIONS = 3  # x, y, and z (always 3 for consistency)


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
            for i in range(DIMENSIONS):
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

    @staticmethod
    def filter_repeated(trajectory: np.ndarray) -> np.ndarray:
        """
        Removes consecutively repeated directions.
        :param trajectory:
        :return:
        """
        filtered = []
        last = None
        for direction in trajectory:
            if not np.array_equal(direction, last):
                filtered.append(direction)
            last = direction
        return np.array(filtered)

    @staticmethod
    def filter_stationary(trajectory: np.ndarray) -> np.ndarray:
        """
        Removes completely stationary directions.
        :param trajectory:
        :return:
        """
        filtered = []
        zeros = np.zeros(DIMENSIONS)
        for direction in trajectory:
            if not np.array_equal(direction, zeros):
                filtered.append(direction)
        return np.array(filtered)

    @staticmethod
    def from_flat(flat: np.ndarray) -> np.ndarray:
        """
        Converts a flat trajectory to a 2d numpy array removing NaNs.
        :param flat: flat trajectory
        :return: 2d numpy array
        """
        reshaped = flat.reshape(-1, DIMENSIONS)
        # shorten if found NaNs (they appear when converting to pandas)
        for i in range(len(reshaped)):
            if np.isnan(reshaped[i]).all():
                return reshaped[:i]
        return reshaped
