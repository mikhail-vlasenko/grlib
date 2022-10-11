from src.grlib.exceptions import NoHandDetectedException
from src.grlib.feature_extraction.pipeline import Pipeline
from src.grlib.load_data.dynamic_gesture_loader import DynamicGestureLoader
import cv2.cv2 as cv
import numpy as np

from src.grlib.dynamic_detection import DynamicDetector
from src.grlib.trajectory.trajectory_classifier import TrajectoryClassifier

ZERO_PRECISION = 0.1

if __name__ == '__main__':
    num_hands = 2
    pipeline = Pipeline(num_hands)
    pipeline.add_stage(0, 0)

    loader = DynamicGestureLoader(
        pipeline, 'left_right_dataset', trajectory_zero_precision=ZERO_PRECISION, key_frames=3
    )
    loader.create_landmarks()

    landmarks = loader.load_landmarks()
    trajectories = loader.load_trajectories()
    x_traj = trajectories.iloc[:, :-1]
    y = np.array(trajectories['label'])

    trajectory_classifier = TrajectoryClassifier()
    trajectory_classifier.fit(np.array(x_traj), y)

    start_shapes = loader.get_start_shape(landmarks, num_hands)

    detector = DynamicDetector(
        start_shapes,
        y,
        pipeline,
        start_pos_confidence=0.1,
        trajectory_classifier=trajectory_classifier,
        update_candidates_every=5,
        candidate_zero_precision=ZERO_PRECISION,
    )

    camera = cv.VideoCapture(0)
    font = cv.FONT_HERSHEY_SIMPLEX
    frame_cnt = 0
    while True:
        ret, frame = camera.read()
        if not ret:
            continue
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        try:
            possible = detector.analyze_frame(frame)
            cv.putText(frame, f'Possible: {possible}', (10, 450), font, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(frame, f'Prediction: {detector.last_pred}', (10, 500), font, 1, (0, 255, 0), 2, cv.LINE_AA)

            cv.imshow('Frame', frame)
        except NoHandDetectedException as e:
            cv.putText(frame, 'No hand detected', (10, 450), font, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(frame, f'Prediction: {detector.last_pred}', (10, 500), font, 1, (0, 255, 0), 2, cv.LINE_AA)

            cv.imshow('Frame', frame)

        # print('\r' + str(pipeline), end='')
