import os
from typing import List

import numpy as np
import pandas as pd

from ..feature_extraction.pipeline import Pipeline
from ..load_data.base_loader import BaseLoader
from ..trajectory.general_direction_builder import GeneralDirectionBuilder
from ..trajectory.key_frames import extract_key_frames


class DynamicGestureLoader(BaseLoader):
    """
    Retrieves landmarks from folder with images.
    """
    def __init__(
            self,
            pipeline: Pipeline,
            path: str,
            verbose: bool = True,
            key_frames: int = 4,
            trajectory_dimensions: int = 3,
            frame_set_separator: str = "_",
            output_trajectory_name: str = "trajectories.csv"
    ):
        """
        :param path: path to dataset's main folder
        """
        super().__init__(pipeline, path, verbose)
        self.key_frames = key_frames
        self.trajectory_dimensions = trajectory_dimensions
        self.frame_set_separator = frame_set_separator
        self.output_trajectory_name = output_trajectory_name
        self.trajectory_builder = GeneralDirectionBuilder(self.trajectory_dimensions)

    def create_landmarks(self, output_file='landmarks.csv'):
        """
        Processes images of gestures and saves results to csv.
        Images are labelled with their folder's name.
        takes a while
        :param output_file: the file path of the file to write to
        :return: None
        """
        landmarks_results = []
        trajectory_results = []
        class_labels = []

        data_labels = [
            folder
            for folder in os.listdir(self.path)
            if os.path.isdir(self.path + folder)
        ]
        for folder in data_labels:
            curr_path = self.path + folder + '/'
            print(f'Processing {curr_path}')

            files: List[str] = [file for file in os.listdir(curr_path)]

            # sorting file names alphabetically will "group" them by prefix
            files.sort()

            prefix = None
            i = 0
            while i < len(files):
                gesture_landmarks = []
                # capture one gesture instance
                while files[i].split(self.frame_set_separator)[0] == prefix:
                    landmarks = self.create_landmarks_for_image(curr_path + files[i])
                    # append only if recognized
                    if len(landmarks) > 0:
                        landmarks = np.array(landmarks)
                        gesture_landmarks.append(landmarks)
                    i += 1

                # save the extracted info
                key_landmarks: List[np.ndarray] = extract_key_frames(gesture_landmarks, self.key_frames)
                trajectory = self.trajectory_builder.make_trajectory(key_landmarks)

                hand_shape_encoding = np.array([], dtype=float)
                for lm in key_landmarks:
                    hand_shape_encoding = np.concatenate((hand_shape_encoding, lm), axis=None)

                landmarks_results.append(hand_shape_encoding)
                trajectory_results.append(trajectory.to_np())
                # append folder name as class label
                class_labels.append(folder)

                # guaranteed to go into the inner while on the next iteration
                prefix = files[i].split(self.frame_set_separator)[0]

        # covert to pandas
        hand_shape_df = pd.DataFrame(landmarks_results)
        trajectory_df = pd.DataFrame(trajectory_results)
        labels_df = pd.DataFrame(class_labels)
        labels_df.columns = ['label']

        hand_shape_df = hand_shape_df.join(labels_df)
        trajectory_df = trajectory_df.join(labels_df)

        # save in csv
        hand_shape_df.to_csv(self.path + output_file, index=False)
        trajectory_df.to_csv(self.path + self.output_trajectory_name, index=False)
