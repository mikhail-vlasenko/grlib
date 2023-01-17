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
    def __init__(self, zero_precision: float = 0.1, use_scaled_zero_precision: bool = True):
        """

        :param zero_precision: how much is considered "no movement on the axis"
        :param use_scaled_zero_precision: if True, the zero precision is scaled by the
        maximal displacement over an axis, but remains at least equal to self.zero_precision.
        This mitigates the problem of the zero precision being too small for fast movements.
        """
        self.zero_precision = zero_precision
        self.use_scaled_zero_precision = use_scaled_zero_precision

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
            current = MediaPipe.hands_spacial_position(landmark)[hand_num]
            directions = self.make_step_directions(last, current, self.zero_precision,
                                                   self.use_scaled_zero_precision)
            trajectory.append(directions)
            last = current
        return np.array(trajectory)

    @staticmethod
    def make_step_directions(previous: np.ndarray, current: np.ndarray,
                             zero_precision: float, use_scaled_zero_precision: bool) -> np.ndarray:
        """
        Creates the directions for a single step
        :param previous: the previous position
        :param current: the current position
        :param zero_precision: how much is considered "no movement on the axis"
        :param use_scaled_zero_precision: if True, the zero precision is scaled.
        :return: the directions for the step
        """
        directions = []
        if use_scaled_zero_precision:
            # increase zero precision if the hand moved a lot
            max_displacement = np.max(np.abs(current - previous))
            if max_displacement > zero_precision * 2:
                zero_precision = max_displacement / 2

        for i in range(DIMENSIONS):
            lower_boundary = previous[i] - zero_precision
            upper_boundary = previous[i] + zero_precision
            if lower_boundary > current[i]:
                directions.append(Direction.DOWN.value)
            elif upper_boundary < current[i]:
                directions.append(Direction.UP.value)
            else:
                directions.append(Direction.STATIONARY.value)
        return np.array(directions)

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
