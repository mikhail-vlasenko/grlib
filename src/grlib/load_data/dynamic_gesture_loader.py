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
            key_frames: int = 3,
            trajectory_zero_precision: float = 0.02,
            # trajectory_dimensions: int = 3,
            frame_set_separator: str = "_",
            output_trajectory_name: str = "trajectories.csv"
    ):
        """
        :param path: path to dataset's main folder
        """
        super().__init__(pipeline, path, verbose)
        self.key_frames = key_frames
        self.trajectory_zero_precision = trajectory_zero_precision
        # self.trajectory_dimensions = trajectory_dimensions
        self.frame_set_separator = frame_set_separator
        self.output_trajectory_name = output_trajectory_name
        self.trajectory_builder = GeneralDirectionBuilder(self.trajectory_zero_precision)

    def create_landmarks(self, output_file='landmarks.csv'):
        """
        Processes images of gestures and saves shapes and trajectories to csv.
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

            i = 0
            while i < len(files):
                # guaranteed to go into the inner loop
                prefix = self._extract_prefix(files[i])

                gesture_image_landmarks = []
                gesture_world_landmarks = []
                # capture one gesture instance
                while i < len(files) and self._extract_prefix(files[i]) == prefix:
                    # extract both types of landmarks, as the first are necessary for trajectory,
                    # and the second are necessary for hand shape recognition
                    image_landmarks = self.create_landmarks_for_image(
                        curr_path + files[i],
                        world_landmarks=False
                    )
                    world_landmarks = self.create_landmarks_for_image(
                        curr_path + files[i],
                        world_landmarks=True
                    )
                    # append only if recognized
                    if len(image_landmarks) > 0:
                        gesture_image_landmarks.append(np.array(image_landmarks))
                        gesture_world_landmarks.append(np.array(world_landmarks))
                    i += 1

                # extract key frames using image-relative landmarks
                key_indices: List[int] = extract_key_frames(gesture_image_landmarks, self.key_frames)
                key_image_landmarks = []
                key_world_landmarks = []
                for k in key_indices:
                    key_image_landmarks.append(gesture_image_landmarks[k])
                    key_world_landmarks.append(gesture_world_landmarks[k])

                # trajectory needs image-relative
                trajectory = self.trajectory_builder.make_trajectory(key_image_landmarks)

                # hand shape needs hand-centered
                hand_shape_encoding = np.array([], dtype=float)
                for lm in key_world_landmarks:
                    hand_shape_encoding = np.concatenate((hand_shape_encoding, lm), axis=None)

                landmarks_results.append(hand_shape_encoding)
                trajectory_results.append(trajectory.to_np())
                # append folder name as class label
                class_labels.append(folder)

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

    def load_trajectories(self, file=None) -> pd.DataFrame:
        """
        Read trajectories from csv file.
        :param file: path to trajectories file, without loader root path.
        Defaults to self.output_trajectory_name.
        :return: the dataframe
        """
        if file is None:
            file = self.output_trajectory_name
        return pd.read_csv(self.path + file)

    @staticmethod
    def get_start_shape(hand_shape_df, num_hands):
        """
        Provides a part of the dataframe that has starting shapes of the hands
        :return: the part of the dataframe
        """
        return hand_shape_df.iloc[:, :(63 * num_hands)]

    def _extract_prefix(self, file):
        return file.split(self.frame_set_separator)[0]
