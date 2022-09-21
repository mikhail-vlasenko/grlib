import numpy as np

from src.grlib.trajectory.general_direction_builder import Direction


class TrajectoryCandidate:
    def __init__(self, target, prediction_class, init_pos, zero_precision, start_timestamp: float):
        """

        :param target: what the trajectory should look like
        :param prediction_class: corresponding class
        :param init_pos:
        :param zero_precision: how much is considered enough movement
            (should correspond to GeneralDirectionBuilder.zero_precision)
        :param start_timestamp: helps to determine when the candidate is too old
        """
        self.target = target
        self.pred_class = prediction_class
        self.position = init_pos
        self.zero_precision = zero_precision
        self.timestamp = start_timestamp

        self.axi = 2

        self.valid = False
        # todo: allow more movement along the same axis? (in 2 consecutive calls of update),
        #   (only if its correct direction)

    def update(self, position) -> bool:
        """
        Should be called every X frame (5, 10 or something)
        :param position: new hand position
        :return: if the trajectory may still be valid (but not necessarily IS valid)
        """
        for i in range(self.axi):
            lower_boundary = self.position[i] - self.zero_precision
            upper_boundary = self.position[i] + self.zero_precision
            direction = Direction.STATIONARY.value
            if lower_boundary > position[i]:
                direction = Direction.DOWN.value
            elif upper_boundary < position[i]:
                direction = Direction.UP.value

            if direction != self.target[i]:
                return False

        if len(self.target) == 3:
            self.valid = True
        self.target = self.target[3:]
        self.position = np.copy(position)
        return True
