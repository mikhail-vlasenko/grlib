import os
from typing import List

import numpy as np
import pandas as pd
from natsort import natsorted

from ..feature_extraction.mediapipe_landmarks import hands_spacial_position
from ..feature_extraction.pipeline import Pipeline
from ..load_data.base_loader import BaseLoader


class ByFolderLoader(BaseLoader):
    """
    Retrieves landmarks from folder with images.
    Landmarks saving format:
    h1l1x, h1l1y, h1l1z, h1l2x, ..., h2l1x, ..., handedness1, handedness2, ..., label
    where h1l1x stands for x coordinate of the first landmark of the first hand
    """
    def __init__(self, pipeline: Pipeline, path: str, verbose: bool = True):
        """
        :param pipeline: the pipeline to use to augment images
        :param path: path to dataset's main folder
        :param verbose: whether to display pipeline information when running
        """
        super().__init__(pipeline, path, verbose)

    def create_landmarks(self, output_file='landmarks.csv', save_positions=False, include_frames_without_hands=False):
        """
        Processes images of gestures and saves results to csv.
        Images are labelled with their folder's name.
        takes a while
        :param output_file: the file path of the file to write to
        :param save_positions: whether to save the positions of the hands in a separate file
        :param include_frames_without_hands: whether to include frames where no hands were detected
            positions will be saved as -1, -1, -1, landmarks as all zeros
        :return: None
        """

        landmarks: List[np.ndarray] = []
        handednesses: List[np.ndarray] = []
        labels: List[str] = []
        positions = []

        data_labels = [folder for folder in os.listdir(self.path) if os.path.isdir(self.path + folder)]
        data_labels = natsorted(data_labels)

        for i, folder in enumerate(data_labels):
            curr_path = self.path + folder + '/'
            print(f'Processing {curr_path}')

            files = [curr_path + file for file in os.listdir(curr_path)]

            files = natsorted(files)

            results: List[(np.ndarray, np.ndarray)] = []
            for file_idx, file in enumerate(files):
                results.append(self.create_landmarks_for_image(file))
                if save_positions:
                    relative_landmarks, _ = self.create_landmarks_for_image(file, world_landmarks=False)
                    if len(relative_landmarks) != 0:
                        positions.append(hands_spacial_position(relative_landmarks).flatten())
                    else:
                        positions.append(np.array([-1, -1, -1]))

            if self.verbose:
                print()

            if not include_frames_without_hands:
                # Remove the instances where no hand was detected (have empty list for landmarks)
                results = [result for result in results if len(result[0]) > 0]
            else:
                # Replace empty list with all zeros
                for j, result in enumerate(results):
                    if len(result[0]) == 0:
                        results[j] = np.zeros((21 * self.pipeline.num_hands * 3)), np.full(self.pipeline.num_hands, -1)

            for result in results:
                landmarks.append(result[0])
                handednesses.append(result[1])
                labels.append(folder)

        df = BaseLoader.make_df_with_handedness(np.array(landmarks), np.array(handednesses), np.array(labels))
        df.to_csv(self.path + output_file, index=False)
        pos_df = pd.DataFrame(positions)
        pos_df.to_csv(self.path + 'positions.csv', index=False)
