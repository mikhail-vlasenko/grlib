# Attempt to import grlib as a dependency, if that fails, try to assume it is a local project
try:
    from grlib.exceptions import NoHandDetectedException
    from grlib.feature_extraction.mediapipe_landmarks import hands_spacial_position
    from grlib.feature_extraction.pipeline import Pipeline
    from grlib.load_data.dynamic_gesture_loader import DynamicGestureLoader
    from grlib.trajectory.general_direction_builder import GeneralDirectionBuilder
except ImportError as ex:
    from src.grlib.exceptions import NoHandDetectedException
    from src.grlib.feature_extraction.mediapipe_landmarks import hands_spacial_position
    from src.grlib.feature_extraction.pipeline import Pipeline
    from src.grlib.load_data.dynamic_gesture_loader import DynamicGestureLoader
    from src.grlib.trajectory.general_direction_builder import GeneralDirectionBuilder

from sklearn.linear_model import LogisticRegression
import cv2.cv2 as cv
import numpy as np
from src.grlib.dynamic_detector import DynamicDetector
from src.grlib.trajectory.trajectory_classifier import TrajectoryClassifier

ZERO_PRECISION = 0.1

if __name__ == '__main__':
    num_hands = 1
    pipeline = Pipeline(num_hands)
    pipeline.add_stage(0, 0)

    # initialize the dataset loader
    loader = DynamicGestureLoader(
        pipeline, '../data/long_trajectories', trajectory_zero_precision=ZERO_PRECISION, key_frames=4
    )
    # this can be commented out after the first run
    # loader.create_landmarks()

    # read the landmarks, handedness and trajectories from the csv files
    landmarks = loader.load_landmarks()
    x_traj, y = loader.load_trajectories()

    # reduce trajectories, i.e. convert [001, 000] to [001]
    x_traj = list(map(GeneralDirectionBuilder.filter_stationary, x_traj))
    x_traj = list(map(GeneralDirectionBuilder.filter_repeated, x_traj))

    trajectory_classifier = TrajectoryClassifier()
    trajectory_classifier.fit(x_traj, y)

    # create models for probabilistic start and end shapes recognition
    start_shapes = loader.get_start_shape(landmarks)
    start_detection_model = LogisticRegression(C=20.0)
    start_detection_model.fit(np.array(start_shapes), y)

    end_shapes = loader.get_end_shape(landmarks)
    end_detection_model = LogisticRegression(C=20.0)
    end_detection_model.fit(np.array(end_shapes), y)

    # initialize the dynamic gesture detector
    detector = DynamicDetector(
        start_detection_model,
        start_pos_confidence=0.25,
        trajectory_classifier=trajectory_classifier,
        update_candidates_every=20,
        candidate_min_time_diff=4,
        verbose=True,
    )

    # initialize the camera
    camera = cv.VideoCapture(0)
    font = cv.FONT_HERSHEY_SIMPLEX
    predicted_gesture = ""
    # continuously read frames from the camera
    while True:
        # get the frame from the camera
        ret, frame = camera.read()
        if not ret:
            continue
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        try:
            # Extract hand information
            landmarks, handedness = pipeline.get_world_landmarks_from_image(frame)
            relative_landmarks, _ = pipeline.get_landmarks_from_image(frame)
            hand_position = hands_spacial_position(relative_landmarks)

            predicted_gesture = detector.get_prediction(landmarks, hand_position[0])
        except NoHandDetectedException as e:
            cv.putText(frame, 'No hand detected', (10, 450), font, 1, (0, 255, 0), 2, cv.LINE_AA)
        finally:
            cv.putText(frame, f'Prediction: {predicted_gesture}', (10, 500), font, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.imshow('Frame', frame)
