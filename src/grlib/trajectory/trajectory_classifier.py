from typing import List, Dict, Union

import numpy as np

from src.grlib.trajectory.general_direction_trajectory import GeneralDirectionTrajectory


class TrajectoryClassifier:
    def __init__(self):
        self.avg_trajectories: Dict[str, np.ndarray] = dict()

    def fit(
            self,
            trajectories: Union[np.ndarray, List[np.ndarray]],
            gestures: Union[np.ndarray, List[str]]
    ):
        # todo: average out the input trajectories, so that we only have a single trajectory per class
        # todo: make use of the GeneralDirectionTrajectory type?
        """
        Remembers correspondence between gestures classes and their trajectories
        :param gestures: list of classification answers that the trajectories correspond to
        :param trajectories:
        :return: None
        """
        assert len(gestures) == len(trajectories)
        temp_gestures: Dict[str, List[np.ndarray]] = dict()
        for i in range(len(trajectories)):
            if gestures[i] not in temp_gestures:
                temp_gestures[gestures[i]] = [trajectories[i]]
            else:
                temp_gestures[gestures[i]].append(trajectories[i])

        for key, value in temp_gestures.items():
            value = np.array(value)
            # averaged trajectories are rounded to valid trajectories (-1, 0, 1 terms)
            self.avg_trajectories[key] = np.array(np.around(value.mean(axis=0)), dtype=int)

    def predict(self, given_trajectory) -> List[str]:
        """
        Lists the gestures that are known to follow the given trajectory
        :param given_trajectory: the trajectory
        :return: list of gesture classes
        """
        # TODO: O(n * traj_len) predict is not quick
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
