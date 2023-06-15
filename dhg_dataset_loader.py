import os
from typing import List

import numpy as np
import pandas as pd
from natsort import natsorted

from convert_dhg_landmarks import remove_palm_center, recenter_landmarks
from src.grlib.feature_extraction.pipeline import Pipeline
from src.grlib.load_data.base_loader import BaseLoader
from src.grlib.trajectory.general_direction_builder import GeneralDirectionBuilder
from src.grlib.trajectory.key_frames import extract_key_frames


def get_file_path(gesture, finger, subject, trial):
    return f'data/DHG2016/gesture_{gesture}/finger_{finger}/subject_{subject}/essai_{trial}/skeleton_world.txt'


trajectory_builder = GeneralDirectionBuilder(0.02)


def extract_info(file_path):
    img_landmarks = np.loadtxt(file_path)

    img_landmarks = remove_palm_center(img_landmarks)

    world_landmarks = recenter_landmarks(img_landmarks)

    key_indices: List[int] = extract_key_frames(img_landmarks, 3)
    key_image_landmarks = []
    key_world_landmarks = []
    for k in key_indices:
        key_image_landmarks.append(img_landmarks[k])
        key_world_landmarks.append(world_landmarks[k])

    trajectory = trajectory_builder.make_trajectory(key_image_landmarks)

    hand_shape_encoding = np.array([], dtype=float)
    for lm in key_world_landmarks:
        hand_shape_encoding = np.concatenate((hand_shape_encoding, lm), axis=None)

    return hand_shape_encoding, trajectory.flatten()


for gesture in range(1, 15):
    for finger in range(1, 3):
        for subject in range(1, 21):
            for trial in range(1, 6):
                file_path = get_file_path(gesture, finger, subject, trial)
                if not os.path.exists(file_path):
                    raise ValueError(f'File {file_path} does not exist')

                hand_shape_encoding, trajectory = extract_info(file_path)
                print(trajectory)
