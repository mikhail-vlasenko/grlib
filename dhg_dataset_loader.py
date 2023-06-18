import os
from typing import List

import numpy as np
import pandas as pd
from natsort import natsorted
from sklearn.linear_model import LogisticRegression

from convert_dhg_landmarks import remove_palm_center, recenter_landmarks
from src.grlib.feature_extraction.mediapipe_landmarks import hands_spacial_position
from src.grlib.feature_extraction.pipeline import Pipeline
from src.grlib.load_data.base_loader import BaseLoader
from src.grlib.trajectory.general_direction_builder import GeneralDirectionBuilder
from src.grlib.trajectory.key_frames import extract_key_frames
from src.grlib.dynamic_detector import DynamicDetector
from src.grlib.trajectory.trajectory_classifier import TrajectoryClassifier
from src.grlib.load_data.dynamic_gesture_loader import DynamicGestureLoader


def get_file_path(gesture, finger, subject, trial):
    return f'data/DHG2016/gesture_{gesture}/finger_{finger}/subject_{subject}/essai_{trial}/skeleton_world.txt'


# only use whole hand gestures
finger = 2

key_frames = 3
zero_precision = 0.02
trajectory_builder = GeneralDirectionBuilder(zero_precision)


def extract_info(file_path):
    img_landmarks = np.loadtxt(file_path)

    img_landmarks = remove_palm_center(img_landmarks)
    world_landmarks = recenter_landmarks(img_landmarks)

    key_indices: List[int] = extract_key_frames(img_landmarks, key_frames)
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


def make_dataset():
    landmarks_results = []
    trajectory_results = []
    handedness_results = []
    class_labels = []

    for gesture in range(1, 15):
        for subject in range(1, 21):
            for trial in range(1, 6):
                file_path = get_file_path(gesture, finger, subject, trial)
                if not os.path.exists(file_path):
                    raise ValueError(f'File {file_path} does not exist')

                hand_shape_encoding, trajectory = extract_info(file_path)
                landmarks_results.append(hand_shape_encoding)
                trajectory_results.append(trajectory)
                handedness_results.append(0)
                # append folder name as class label
                class_labels.append(gesture)

    hand_shape_df = pd.DataFrame(landmarks_results)
    trajectory_df = pd.DataFrame(trajectory_results)
    handedness_df = pd.DataFrame(handedness_results)
    handedness_df.columns = [f'handedness {i}' for i in range(1, len(handedness_df.columns)+1)]

    labels_df = pd.DataFrame(class_labels)
    labels_df.columns = ['label']

    hand_shape_df = hand_shape_df.join(handedness_df)
    hand_shape_df = hand_shape_df.join(labels_df)
    trajectory_df = trajectory_df.join(labels_df)

    # save in csv
    hand_shape_df.to_csv('data/DHG2016/landmarks.csv', index=False)
    trajectory_df.to_csv('data/DHG2016/trajectories.csv', index=False)


def test_on_dhg():
    # make_dataset()
    # print('Dataset created')
    landmarks = pd.read_csv('data/DHG2016/landmarks.csv')

    df = pd.read_csv('data/DHG2016/trajectories.csv')
    trajectories = []
    data = df.iloc[:, :-1].values
    for t in data:
        trajectories.append(GeneralDirectionBuilder.from_flat(t))
    x_traj, y = trajectories, np.array(df['label'])

    x_traj = list(map(GeneralDirectionBuilder.filter_stationary, x_traj))
    x_traj = list(map(GeneralDirectionBuilder.filter_repeated, x_traj))

    trajectory_classifier = TrajectoryClassifier(allow_multiple=True, popularity_threshold=0.2)
    trajectory_classifier.fit(x_traj, y)

    # create models for probabilistic start and end shapes recognition
    start_shapes = DynamicGestureLoader.get_start_shape(landmarks)
    start_detection_model = LogisticRegression(C=20.0, max_iter=1000)
    start_detection_model.fit(np.array(start_shapes), y)

    end_shapes = DynamicGestureLoader.get_end_shape(landmarks)
    end_detection_model = LogisticRegression(C=20.0, max_iter=1000)
    end_detection_model.fit(np.array(end_shapes), y)

    # initialize the dynamic gesture detector
    detector = DynamicDetector(
        start_detection_model,
        start_pos_confidence=0.1,
        trajectory_classifier=trajectory_classifier,
        update_candidates_every=20,
        candidate_min_time_diff=4,
        verbose=False,
    )

    correct_predictions = 0
    total_predictions = 0
    total_gestures = 0
    for gesture in range(1, 15):
        for subject in range(1, 21):
            for trial in range(1, 6):
                total_gestures += 1
                file_path = get_file_path(gesture, finger, subject, trial)
                # Input data, each row corresponds to a time point
                landmarks = np.loadtxt(file_path)
                landmarks = landmarks.reshape((landmarks.shape[0], 22, 3))

                img_landmarks = remove_palm_center(landmarks)
                world_landmarks = recenter_landmarks(img_landmarks)
                world_landmarks = world_landmarks.reshape((world_landmarks.shape[0], 63))

                prediction = None
                for i in range(len(landmarks)):
                    hand_position = hands_spacial_position(img_landmarks[i])
                    predicted_gesture = detector.get_prediction(world_landmarks[i], hand_position[0])
                    if predicted_gesture is not None:
                        prediction = predicted_gesture
                        break

                if prediction is None:
                    # fill the end with the final frame (as if the hand has stopped moving)
                    # to give the detector a chance to detect the end
                    for i in range(20):
                        hand_position = hands_spacial_position(img_landmarks[-1])
                        predicted_gesture = detector.get_prediction(world_landmarks[-1], hand_position[0])
                        if predicted_gesture is not None:
                            prediction = predicted_gesture
                            break

                if prediction is not None:
                    total_predictions += 1
                    if prediction == gesture:
                        correct_predictions += 1

    print(f'Correct from predictions: {correct_predictions / total_predictions}')
    print(f'Correct from given gestures: {correct_predictions / total_gestures}')


if __name__ == '__main__':
    test_on_dhg()
