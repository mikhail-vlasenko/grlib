import numpy as np

from src.grlib.trajectory.general_direction_builder import Direction


class TrajectoryCandidate:
    """
    Tracks a path of the hand for a specific dynamic gesture.
    If the path taken corresponds to the `target` path, the dynamic gesture is considered `valid`.
    """
    def __init__(self, target: np.ndarray, prediction_class, init_pos, zero_precision, start_timestamp: float):
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

        self.used_axi = {"x": True, "y": True, "z": False}

        self.valid = False
        # todo: allow more movement along the same axis? (in 2 consecutive calls of update),
        #   (only if its correct direction)

    def update(self, position) -> bool:
        """
        The remembered hand position (`self.position`) is compared to the most recent hand position.
        If the direction matches whatever is necessary for the gesture,
        the candidate's remaining directions (`self.target`) are reduced.
        Should be called every X frame (5, 10 or something)
        :param position: new hand position
        :return: if the trajectory may still be valid (but not necessarily IS valid)
        """
        for i, a in enumerate(["x", "y", "z"]):
            # only update if the axis is used
            if self.used_axi[a]:
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

    def __repr__(self):
        return f'Candidate for {self.pred_class}: target={self.target}, position={self.position}'
