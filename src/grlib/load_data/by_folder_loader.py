import os
from typing import List

import numpy as np
import pandas as pd

from ..feature_extraction.pipeline import Pipeline
from ..load_data.base_loader import BaseLoader


class ByFolderLoader(BaseLoader):
    """
    Retrieves landmarks from folder with images.
    Landmarks saving format:
    h1l1x, h1l1y, h1l1z, h1l2x, ..., h2l1x, ..., handedness1, handedness2, ..., label
    where h1l1x stands for x coordinate of the first landmark of the first hand
    """
    def __init__(self, pipeline: Pipeline, path: str, max_hands: int, verbose: bool = True):
        """
        :param pipeline: the pipeline to use to augment images
        :param path: path to dataset's main folder
        :param verbose: whether to display pipeline information when running
        """
        super().__init__(pipeline, path, verbose)
        self.max_hands = max_hands

    def create_landmarks(self, output_file='landmarks.csv'):
        """
        Processes images of gestures and saves results to csv.
        Images are labelled with their folder's name.
        takes a while
        :param output_file: the file path of the file to write to
        :return: None
        """

        landmarks: List[np.ndarray] = []
        handednesses: List[np.ndarray] = []
        labels: List[str] = []

        data_labels = [folder for folder in os.listdir(self.path) if os.path.isdir(self.path + folder)]

        for i, folder in enumerate(data_labels):
            curr_path = self.path + folder + '/'
            print(f'Processing {curr_path}')

            files = [curr_path + file for file in os.listdir(curr_path)]

            results: List[(np.ndarray, np.ndarray)] = []
            for file_idx, file in enumerate(files):
                results.append(self.create_landmarks_for_image(file))

            if self.verbose:
                print()

            # Remove the instances where no hand was detected (have empty list for landmarks)
            results = [result for result in results if len(result[0]) > 0]

            for result in results:
                landmarks.append(result[0])
                handednesses.append(result[1])
                labels.append(folder)

        df1 = pd.DataFrame(np.array(landmarks))
        df2 = pd.DataFrame(np.array(handednesses))
        df2.columns = [f'handedness {i}' for i in range(1, self.max_hands+1)]
        df = pd.concat([df1, df2], axis=1)
        df['label'] = np.array(labels)

        df.to_csv(self.path + output_file, index=False)
