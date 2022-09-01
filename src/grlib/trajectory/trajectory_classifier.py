from typing import List, Dict, Union

import numpy as np

from src.grlib.trajectory.general_direction_trajectory import GeneralDirectionTrajectory


class TrajectoryClassifier:
    def __init__(self):
        self._state: Dict[np.ndarray, List[str]] = dict()

    def fit(
            self,
            trajectories: Union[np.ndarray, List[np.ndarray]],
            gestures: Union[np.ndarray, List[str]]
    ):
        # todo: average out the input trajectories, so that we only have a single trajectory per class
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

        avg_trajectories: Dict[str, np.ndarray] = dict()
        for key, value in temp_gestures.items():
            value = np.array(value)
            avg_trajectories[key] = value.mean(axis=0)

        for key, value in avg_trajectories.items():
            if value not in self._state:
                self._state[value] = [key]
            else:
                self._state[value].append(key)

    def predict(self, trajectory) -> List[str]:
        """
        Lists the gestures that are known to follow the given trajectory
        :param trajectory: the trajectory
        :return: list of gesture classes
        """
        return self._state[trajectory]
