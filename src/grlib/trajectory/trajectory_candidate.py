import numpy as np

from src.grlib.exceptions import UpdateOnFinishedCandidateException
from src.grlib.trajectory.general_direction_builder import Direction, GeneralDirectionBuilder


class TrajectoryCandidate:
    """
    Tracks a path of the hand for a specific dynamic gesture.
    If the path taken corresponds to the `target` path, the dynamic gesture is considered `valid`.
    """
    def __init__(self,
                 target: np.ndarray,
                 prediction_class,
                 init_pos,
                 zero_precision: float,
                 use_scaled_zero_precision: bool,
                 start_timestamp: float,
                 used_axi: dict = None):
        """

        :param target: what the trajectory should look like
        :param prediction_class: corresponding class
        :param init_pos: initial position of the hand
        :param zero_precision: how much is considered enough movement
            (should correspond to GeneralDirectionBuilder.zero_precision)
        :param use_scaled_zero_precision: if True, the zero precision is scaled
        like in the GeneralDirectionBuilder.
        :param start_timestamp: helps to determine when the candidate is too old
        :param used_axi: which axi are taken into account for the trajectory. Defaults to x and y.
        """
        self.target = target
        self.pred_class = prediction_class
        self.position = init_pos
        self.zero_precision = zero_precision
        self.use_scaled_zero_precision = use_scaled_zero_precision
        self.timestamp = start_timestamp

        if used_axi is None:
            self.used_axi = {"x": True, "y": True, "z": False}
        else:
            self.used_axi = used_axi

        # if the trajectory is complete
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
        if self.valid:
            raise UpdateOnFinishedCandidateException("Candidate is already valid")

        directions = GeneralDirectionBuilder.make_step_directions(
            self.position, position, self.zero_precision, self.use_scaled_zero_precision)

        for i, a in enumerate(["x", "y", "z"]):
            # only update if the axis is used
            if self.used_axi[a]:
                if directions[i] != self.target[0][i]:
                    return False

        if len(self.target) == 1:
            self.valid = True
        # remove one set of directions
        self.target = self.target[1:]
        self.position = np.copy(position)
        return True

    def __repr__(self):
        return f'Candidate for {self.pred_class}: target={self.target}, position={self.position}'
