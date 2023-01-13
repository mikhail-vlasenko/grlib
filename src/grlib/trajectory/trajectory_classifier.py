from typing import List, Dict, Union

import numpy as np
from collections import Counter


class TrajectoryClassifier:
    def __init__(self, allow_multiple=True, aggregate="most common", verbose=False):
        """

        :param allow_multiple: whether to allow multiple trajectories for the same gesture.
        :param aggregate: how to aggregate multiple trajectories for the same gesture.
        Options: ["most common", "average"]
        :param verbose:
        """
        # stores gesture to trajectory(ies) correspondence
        self.avg_trajectories: Dict[str, np.ndarray] = dict()
        # todo
        self.allow_multiple = allow_multiple
        self.aggregate = aggregate
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

        if self.aggregate == "average":
            # todo
            for key, value in temp_gestures.items():
                value = np.array(value)
                # averaged trajectories are rounded to valid trajectories (-1, 0, 1 terms)
                self.avg_trajectories[key] = np.array(np.around(value.mean(axis=0)), dtype=int)
        elif self.aggregate == "most common":
            for key, value in temp_gestures.items():
                # convert to string so it can be hashed
                counts = Counter(arr.tostring() for arr in value)
                most_common_arr = np.fromstring(counts.most_common(1)[0][0]).reshape(value[0].shape)
                self.avg_trajectories[key] = most_common_arr

        if self.verbose:
            print(f'trajectories: {self.avg_trajectories}')

    def predict(self, given_trajectory) -> List[str]:
        """
        Lists the gestures that are known to follow the given trajectory
        :param given_trajectory: the trajectory
        :return: list of gesture classes
        """
        ans = []
        for pred_class, trajectory in self.avg_trajectories.items():
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
