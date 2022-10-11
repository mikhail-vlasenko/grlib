from typing import Tuple

import numpy as np
import pandas as pd

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
        self.representatives = {}
        self.construct_representatives()

    def construct_representatives(self):
        """
        Builds the representatives of each class. They are constructed by taking the mean of all landmarks
        within a class.
        """
        groups = self.dataset.groupby('label')
        for name, group in groups:
            np_array = group.drop('label', axis=1).to_numpy()
            representative = np.mean(np_array, axis=0)
            self.representatives[name] = representative

    def closest_representative(self, sample: np.ndarray) -> Tuple[float, int]:
        """
        Finds the closest representative to the sample using the specified similarity metric.
        :param sample: the sample to find the closest representative of
        :return: tuple of similarity score and the class closest to the sample
        """
        max_similarity = -float('inf')
        max_class = -1
        for label, representative in self.representatives.items():
            similarity = FalsePositiveFilter.cosine_similarity(sample, representative) if self.metric == 'cosine' \
                else FalsePositiveFilter.euclidean_similarity(sample, representative)
            if similarity >= max_similarity:
                max_similarity = similarity
                max_class = label
        return max_similarity, max_class

    def is_relevant(self, sample: np.array) -> bool:
        """
        Checks if a sample is relevant, i.e.: if it is close enough to any class.
        :param sample: the sample to check if is relevant
        :return: True if sample is close enough to any class, False otherwise
        """
        max_similarity, _ = self.closest_representative(sample)

        if max_similarity >= self.confidence:
            return True
        return False

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


