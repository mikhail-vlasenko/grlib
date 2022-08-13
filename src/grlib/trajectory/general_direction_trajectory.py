from dataclasses import dataclass

import numpy as np


@dataclass
class GeneralDirectionTrajectory:
    length: int
    x_directions: np.ndarray
    y_directions: np.ndarray
    z_directions: np.ndarray

    def to_np(self) -> np.ndarray:
        return np.concatenate((self.x_directions, self.y_directions, self.z_directions), axis=0)
