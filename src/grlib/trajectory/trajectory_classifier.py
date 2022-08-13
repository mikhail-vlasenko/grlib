from typing import List, Dict

from src.grlib.trajectory.general_direction_trajectory import GeneralDirectionTrajectory


class TrajectoryClassifier:
    def __init__(self):
        self._state: Dict[GeneralDirectionTrajectory, List[int]] = dict()

    def fit(self, gestures: List, trajectories: List):
        """
        Remembers correspondence between gestures classes and their trajectories
        :param gestures: list of classification answers that the trajectories correspond to
        :param trajectories:
        :return: None
        """
        assert len(gestures) == len(trajectories)
        for i in range(len(trajectories)):
            if trajectories[i] not in self._state:
                self._state[trajectories[i]] = [gestures[i]]
            else:
                self._state[trajectories[i]].append(gestures[i])

    def predict(self, trajectory) -> List:
        """
        Lists the gestures that are known to follow the given trajectory
        :param trajectory: the trajectory
        :return: list of gesture classes
        """
        return self._state[trajectory]
