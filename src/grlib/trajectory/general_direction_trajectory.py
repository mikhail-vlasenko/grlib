from dataclasses import dataclass

import numpy as np


@dataclass
class GeneralDirectionTrajectory:
    length: int
    x_directions: np.ndarray
    y_directions: np.ndarray
    z_directions: np.ndarray

    def to_np(self) -> np.ndarray:
        """
        Coverts x = [1,1]; y = [0,0]; z = [0,0];
        to [1 0 0 1 0 0]
        :return:
        """
        return np.concatenate(
            (
                np.expand_dims(self.x_directions, axis=0),
                np.expand_dims(self.y_directions, axis=0),
                np.expand_dims(self.z_directions, axis=0)
            ), axis=0
        ).T.flatten()
