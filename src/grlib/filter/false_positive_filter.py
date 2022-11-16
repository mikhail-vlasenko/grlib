import dataclasses
from typing import Tuple, Dict

import numpy as np
import pandas as pd


@dataclasses.dataclass
class Representative:
    mean_landmarks: np.ndarray
    # elements can be 0 (left), 1 (right), one element per hand
    handedness: np.ndarray


class FalsePositiveFilter(object):
    """
    Filters out landmarks that are not close enough to any class.
    Works by creating representative samples of each class by taking the mean landmarks of each class.
    Then compares every incoming sample using cosine or euclidean distance to the representative and
    if any representative is within `confidence`, the sample is marked as relevant.
    """

    def __init__(self, dataset: pd.DataFrame, metric: str = 'cosine', confidence: float = 0.9):
        """
        :param dataset: the dataset of all classes. Requires the dataset to have a column 'label', which
            links the sample to the class
        :param metric: the metric to use, can be either 'cosine' or 'euclidean'
        :param confidence: how similar the sample should be to any representative to be marked as relevant
        """
        if metric != 'cosine' and metric != 'euclidean':
            raise ValueError(f'{metric} is not a supported similarity metric.')
        self.dataset = dataset
        self.metric = metric
        self.confidence = confidence
        self.representatives: Dict[str, Representative] = {}
        self.construct_representatives()

    def construct_representatives(self):
        """
        Builds the representatives of each class. They are constructed by taking the mean of all landmarks
        within a class.
        """
        dataset = self.dataset

        handedness_df = pd.DataFrame(self.dataset['label'])
        for col in self.dataset.columns:
            if 'handedness' in col:
                dataset = dataset.drop(col, axis=1)
                handedness_df = pd.concat([handedness_df, self.dataset[col]], axis=1)

        hands = handedness_df.groupby('label')
        handedness = {}
        for name, group in hands:
            np_array = group.drop('label', axis=1).to_numpy()
            handedness[name] = np.median(np_array, axis=0)

        groups = dataset.groupby('label')
        for name, group in groups:
            np_array = group.drop('label', axis=1).to_numpy()
            representative = Representative(np.mean(np_array, axis=0), handedness[name])
            self.representatives[str(name)] = representative

    def closest_representative(
            self,
            sample_landmarks: np.ndarray,
            sample_handedness: np.ndarray
    ) -> Tuple[float, int]:
        """
        Finds the closest representative to the sample using the specified similarity metric.
        The sample can be multiple hands
        :param sample_landmarks: the sample to find the closest representative of
        :param sample_handedness: the left/right info about the sample
        :return: tuple of similarity score and the class closest to the sample
        """
        max_similarity = -float('inf')
        max_class = -1
        for label, representative in self.representatives.items():
            if sample_handedness != representative.handedness:
                # left/right info is not aligned
                continue

            similarity = FalsePositiveFilter.cosine_similarity(
                sample_landmarks, representative.mean_landmarks) if self.metric == 'cosine' \
                else FalsePositiveFilter.euclidean_similarity(sample_landmarks, representative.mean_landmarks)
            if similarity >= max_similarity:
                max_similarity = similarity
                max_class = label
        return max_similarity, max_class

    def is_relevant(self, sample: np.array, sample_handedness: np.ndarray) -> bool:
        """
        Checks if a sample is relevant, i.e.: if it is close enough to any class.
        :param sample: the sample to check if is relevant
        :param sample_handedness: the left/right info about the sample
        :return: True if sample is close enough to any class, False otherwise
        """
        max_similarity, _ = self.closest_representative(sample, sample_handedness)

        if max_similarity >= self.confidence:
            return True
        return False

    def drop_wrong_hands(
            self,
            landmarks: np.ndarray,
            handedness: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        """
        Specifically for 1-hand datasets, this function removes all hands,
        whose handedness, combined with landmark similarity is not suited for the dataset.
        This helps with pictures which have too many hands.
        :param landmarks: landmarks of all hands on the picture
        :param handedness: the left/right info about all hands on the picture
        :return: (reduced) set of landmarks and handednesses
        """
        res_landmarks = []
        res_handedness = []
        n_hands = len(handedness)
        one_hand_len = int(len(landmarks) / n_hands)
        for i in range(n_hands):
            curr_landmarks = landmarks[i * one_hand_len:(i+1) * one_hand_len]
            if self.is_relevant(curr_landmarks, handedness[i]):
                res_landmarks.append(curr_landmarks)
                res_handedness.append(handedness[i])
        return np.array(res_landmarks).flatten(), np.array(res_handedness)

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Computes cosine similarity of two vectors.
        :param v1: the first vector
        :param v2: the second vector
        :return: the cosine similarity of the two vectors
        """
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    @staticmethod
    def euclidean_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Computes euclidean similarity of two vectors. Computed as 1 / (1 + d(v1, v2)), where d(v1, v2) is the euclidean
        distance between the two vectors.
        :param v1: the first vector
        :param v2: the second vector
        :return: the euclidean similarity between the two vectors
        """
        return 1 / (1 + np.sqrt(np.sum((v1 - v2)**2)))


