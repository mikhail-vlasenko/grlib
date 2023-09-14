# Attempt to import grlib as a dependency, if that fails, try to assume it is a local project
import time

from src.grlib.utils.plot_trajectory import plot_trajectory

try:
    from grlib.exceptions import NoHandDetectedException
    from grlib.feature_extraction.mediapipe_landmarks import get_landmarks_at_position, hands_spacial_position
    from grlib.feature_extraction.pipeline import Pipeline
    from grlib.filter.false_positive_filter import FalsePositiveFilter
    from grlib.load_data.dynamic_gesture_loader import DynamicGestureLoader
    from grlib.trajectory.general_direction_builder import GeneralDirectionBuilder
    from grlib.dynamic_detector import DynamicDetector
    from grlib.trajectory.trajectory_classifier import TrajectoryClassifier
except ImportError as ex:
    from src.grlib.exceptions import NoHandDetectedException
    from src.grlib.feature_extraction.mediapipe_landmarks import get_landmarks_at_position, hands_spacial_position
    from src.grlib.feature_extraction.pipeline import Pipeline
    from src.grlib.filter.false_positive_filter import FalsePositiveFilter
    from src.grlib.load_data.dynamic_gesture_loader import DynamicGestureLoader
    from src.grlib.trajectory.general_direction_builder import GeneralDirectionBuilder
    from src.grlib.dynamic_detector import DynamicDetector
    from src.grlib.trajectory.trajectory_classifier import TrajectoryClassifier

import pandas as pd
from sklearn.linear_model import LogisticRegression
import cv2.cv2 as cv
import numpy as np

ZERO_PRECISION = 0.1

if __name__ == '__main__':
    num_hands = 1
    pipeline = Pipeline(num_hands)
    pipeline.add_stage(0, 0)

    # initialize the dataset loader
    loader = DynamicGestureLoader(
        pipeline, '../data/dynamic_dataset', trajectory_zero_precision=ZERO_PRECISION, key_frames=3
    )
    # this can be commented out after the first run
    # loader.create_landmarks()

    # read the landmarks, handedness and trajectories from the csv files
    landmarks = loader.load_landmarks()
    x_traj, y = loader.load_trajectories()

    # Filter out landmarks and trajectories with label 'backward' or 'forward'
    landmarks = landmarks[~landmarks['label'].isin(['backward', 'forward'])]
    mask = ~(np.isin(y, ['backward', 'forward']))
    x_traj = np.array(x_traj)[mask].tolist()
    y = y[mask]

    # # Get the first n entries for each label in the landmarks
    # n = 3
    # landmarks = landmarks.groupby('label').head(n).reset_index(drop=True)
    # labels_df = pd.DataFrame({'label': y})
    # indices_first3 = labels_df.groupby('label').head(n).index.values
    # x_traj = np.array(x_traj)[indices_first3].tolist()
    # y = y[indices_first3]

    # reduce trajectories, i.e. convert [001, 000] to [001]
    x_traj = list(map(GeneralDirectionBuilder.filter_stationary, x_traj))
    x_traj = list(map(GeneralDirectionBuilder.filter_repeated, x_traj))
    x_traj = list(map(GeneralDirectionBuilder.replace_empty, x_traj))

    # print("Using these trajectories:")
    # print(x_traj)

    trajectory_classifier = TrajectoryClassifier()
    trajectory_classifier.fit(x_traj, y)

    # create models for probabilistic start and end shapes recognition
    start_shapes = loader.get_start_shape(landmarks)
    start_detection_model = LogisticRegression(C=20.0)
    start_detection_model.fit(np.array(start_shapes), y)

    end_shapes = loader.get_end_shape(landmarks)
    end_detection_model = LogisticRegression(C=20.0)
    end_detection_model.fit(np.array(end_shapes), y)

    # make a dataframe with all the start shapes, their handedness and labels
    # for the false positive filter (although shapes are irrelevant with confidence -1)
    fp_data = pd.DataFrame(np.array(start_shapes))
    for i in range(1, num_hands + 1):
        name = f'handedness {i}'
        fp_data[name] = landmarks[name]
    fp_data['label'] = y
    # start detection model will decide which hand shapes are close enough to be make candidates,
    # so we only want to filter out by handedness, thus the confidence is set to -1
    fp_filter = FalsePositiveFilter(fp_data, 'cosine', confidence=-1)

    # when running the camera, more hands may be in the frame, so we want to detect all of them
    run_pipeline = Pipeline(1)
    run_pipeline.add_stage(0, 0)

    # initialize the dynamic gesture detector
    detector = DynamicDetector(
        start_detection_model,
        start_pos_confidence=0.125,
        trajectory_classifier=trajectory_classifier,
        update_candidates_every=10,
        candidate_zero_precision=ZERO_PRECISION,
        end_shape_detection_model=end_detection_model,
        end_pos_confidence=0.125,
        verbose=False,
    )

    # initialize the camera
    landmark_df = pd.read_csv('../data/dynamic_dataset_online/landmarks.csv')
    position_df = pd.read_csv('../data/dynamic_dataset_online/positions.csv')

    labels = landmark_df['label']
    POSSIBLE_SEQUENCES = ['triple simple gestures', 'flowing sequence', 'flowing sequence 2',
                          'circle infinity', 'alphabetical order 1']
    sequence_name = POSSIBLE_SEQUENCES[2]
    sequence_start = None
    sequence_end = None
    for i in range(len(labels)):
        if labels[i] == sequence_name and sequence_start is None:
            sequence_start = i
        elif labels[i] != sequence_name and sequence_start is not None:
            sequence_end = i
            break

    if sequence_end is None:
        sequence_end = len(labels)

    # plot_trajectory(position_df.iloc[sequence_start:sequence_end].copy())
    # exit()

    frame_cnt = 0
    last_pred = 0
    predicted_gesture = ""
    all_predictions = []
    t = time.time()
    for i in range(sequence_start, sequence_end):
        # useful to know relative time
        frame_cnt += 1
        try:
            if last_pred < frame_cnt - 5:
                predicted_gesture = ""

            # Extract hand information
            # print(landmarks.iloc[i])
            landmarks = np.array(landmark_df.iloc[i])[:-2]
            handedness = np.array([1])
            hand_position = np.array([position_df.iloc[i]])

            if hand_position[0][0] == -1:
                raise NoHandDetectedException

            # get index of the best hand
            indices = fp_filter.best_hands_indices_silent(landmarks, handedness)
            if len(indices) != 0:
                index = indices[0]
                # select the best hand
                landmarks_best = get_landmarks_at_position(landmarks, index)
                hand_position_best = hand_position[index]
                # add new trajectory candidates
                possible = detector.add_candidates(landmarks_best, hand_position_best)
                # update candidates and get possible classes
                classes = detector.update_candidates(hand_position_best)
                if len(classes) != 0:
                    # print(f'Complete trajectory for: {classes}')
                    prediction = detector.evaluate_end_shape(landmarks_best, classes)
                else:
                    prediction = None

                if prediction is not None:
                    # to display the prediction
                    predicted_gesture = prediction
                    print(f'Prediction: {predicted_gesture} at frame {frame_cnt}')
                    all_predictions.append(predicted_gesture)
                    last_pred = frame_cnt
                    # clean the current candidates because the gesture is found
                    detector.clear_candidates()
            else:
                # update candidates even if no valid hand shape is detected
                # use the first hand for position
                # todo: can use semi-handtracking to get the correct hand
                #  (just the closest position to the last known)
                detector.update_candidates(hand_position[0])
        except NoHandDetectedException as e:
            # detector.update_candidates() is not called, but the frame counter,
            # which acts a measure of time, still need to be updated
            detector.count_frame()
            # todo: clear candidates if no hand is detected for a while?
    print(f'Time: {time.time() - t}. for {frame_cnt} frames')
    print(f'Predictions for {sequence_name}: {str(all_predictions)[1:-1]}'.replace("'", ""))
