from typing import List, Dict, Union

import numpy as np
import math
from collections import Counter

from src.grlib.trajectory.general_direction_builder import DIMENSIONS


class TrajectoryClassifier:
    def __init__(self, allow_multiple=True, popularity_threshold=0.3, verbose=False):
        """

        :param allow_multiple: whether to allow multiple trajectories for the same gesture.
        :param popularity_threshold: what share of the total should a trajectory constitute
        to be considered representative.
        :param verbose: whether to print debug info
        """
        # stores gesture to trajectory(ies) correspondence
        self.repr_trajectories: Dict[str, List[np.ndarray]] = dict()
        self.allow_multiple = allow_multiple
        self.popularity_threshold = popularity_threshold
        self.verbose = verbose

    def fit(
            self,
            trajectories: Union[np.ndarray, List[np.ndarray]],
            gestures: Union[np.ndarray, List[str]],
    ):
        """
        Remembers correspondence between gestures classes and their trajectories
        :param trajectories: the trajectories for classification
        :param gestures: list of classification answers that the trajectories correspond to
        :return: None
        """
        if len(gestures) != len(trajectories):
            raise ValueError("Number of trajectories and gestures must match")
        temp_gestures: Dict[str, List[np.ndarray]] = dict()
        for i in range(len(trajectories)):
            if gestures[i] not in temp_gestures:
                temp_gestures[gestures[i]] = [trajectories[i]]
            else:
                temp_gestures[gestures[i]].append(trajectories[i])

        for key, value in temp_gestures.items():
            # convert to string so it can be hashed
            counts = Counter(arr.tostring() for arr in value)
            if not self.allow_multiple:
                most_common_arr = np.fromstring(
                    counts.most_common(1)[0][0], dtype=int).reshape((-1, DIMENSIONS))
                self.repr_trajectories[key] = [most_common_arr]
            else:
                # with such popularity, at most that many trajectories will be stored
                most_considered = math.floor(1 / self.popularity_threshold)
                most_common = counts.most_common(most_considered)
                most_common_trajectories = []
                for t in most_common:
                    most_common_trajectories.append(np.fromstring(
                        t[0], dtype=int).reshape((-1, DIMENSIONS)))
                self.repr_trajectories[key] = most_common_trajectories

        if self.verbose:
            print(f'trajectories: {self.repr_trajectories}')

    def get_trajectories(self, gesture: str) -> List[np.ndarray]:
        """
        Returns the trajectories that correspond to the given gesture
        :param gesture: the gesture
        :return: list of trajectories
        """
        return self.repr_trajectories[gesture]

    def predict(self, given_trajectory) -> List[str]:
        """
        Lists the gestures that are known to follow the given trajectory
        :param given_trajectory: the trajectory
        :return: list of gesture classes
        """
        ans = []
        for pred_class, trajectory in self.repr_trajectories.items():
            if self.eq_trajectories(given_trajectory, trajectory):
                ans.append(pred_class)
        return ans

    @staticmethod
    def eq_trajectories(traj1, traj2) -> bool:
        """
        Determines if the trajectories are close enough to be considered equal
        :param traj1:
        :param traj2:
        :return:
        """
        assert len(traj1) == len(traj2)
        for i in range(len(traj1)):
            if traj1[i] != traj2[i]:
                return False
        return True
