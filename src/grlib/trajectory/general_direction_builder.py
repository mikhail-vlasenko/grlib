from enum import Enum


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
    def __init__(self, dimensions=2, zero_precision=0.1):
        """

        :param dimensions:
        :param zero_precision: how much is considered "no movement on the axis"
        """
        self.dimensions = dimensions
        self.zero_precision = zero_precision

    def make_trajectory(self, landmark_sequence):
        """
        Creates the trajectory
        :param landmark_sequence: a sequence of landmarks from (sample) frames
        :return: the trajectory encoding as an array
        """
        trajectory = []
        last = self.hand_center(landmark_sequence[0])
        for landmark in landmark_sequence[1:]:
            new_point = []
            for i in range(self.dimensions):
                current = self.hand_center(landmark)
                lower_boundary = last[i] - self.zero_precision
                upper_boundary = last[i] + self.zero_precision
                if lower_boundary > current[i]:
                    new_point.append(Direction.DOWN)
                elif upper_boundary < current[i]:
                    new_point.append(Direction.UP)
                else:
                    new_point.append(Direction.STATIONARY)
            trajectory.append(new_point)
        return trajectory

    def hand_center(self, hand_landmarks):
        return hand_landmarks[0]
