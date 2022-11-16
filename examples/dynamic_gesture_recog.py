import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.grlib.exceptions import NoHandDetectedException
from src.grlib.feature_extraction.pipeline import Pipeline
from src.grlib.filter.false_positive_filter import FalsePositiveFilter
from src.grlib.load_data.dynamic_gesture_loader import DynamicGestureLoader
import cv2.cv2 as cv
import numpy as np

from src.grlib.dynamic_detection import DynamicDetector
from src.grlib.trajectory.trajectory_classifier import TrajectoryClassifier

ZERO_PRECISION = 0.1

if __name__ == '__main__':
    num_hands = 1
    pipeline = Pipeline(num_hands)
    pipeline.add_stage(0, 0)

    loader = DynamicGestureLoader(
        pipeline, '../src/left_right_dataset', trajectory_zero_precision=ZERO_PRECISION, key_frames=3
    )
    #loader.create_landmarks()

    landmarks = loader.load_landmarks()
    trajectories = loader.load_trajectories()
    x_traj = trajectories.iloc[:, :-1]
    y = np.array(trajectories['label'])

    trajectory_classifier = TrajectoryClassifier()
    trajectory_classifier.fit(np.array(x_traj), y)

    start_shapes = loader.get_start_shape(landmarks, num_hands)
    start_detection_model = LogisticRegression()
    start_detection_model.fit(np.array(start_shapes), y)

    fp_data = pd.DataFrame(np.array(start_shapes))
    # one hand ONLY
    df2 = pd.DataFrame(np.ones(len(np.array(start_shapes))))
    df2.columns = ['handedness 1']
    fp_data = pd.concat([fp_data, df2], axis=1)
    fp_data['label'] = np.array(y)
    fp_filter = FalsePositiveFilter(fp_data, 'cosine', confidence=0.5)

    run_pipeline = Pipeline(3)
    run_pipeline.add_stage(0, 0)

    detector = DynamicDetector(
        start_detection_model,
        y,
        run_pipeline,
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
            landmarks, handedness, hand_position = detector.extract_landmarks(frame)
            print(handedness)
            landmarks, handedness = fp_filter.drop_wrong_hands(landmarks, handedness)
            print(handedness)
            if len(landmarks) != 0:
                possible = detector.analyze_frame(landmarks, hand_position)
                cv.putText(frame, f'Possible: {possible}', (10, 450), font, 1, (0, 255, 0), 2, cv.LINE_AA)
                cv.putText(frame, f'Prediction: {detector.last_pred}', (10, 500), font, 1, (0, 255, 0), 2, cv.LINE_AA)
            else:
                detector.frame_cnt += 1
                raise NoHandDetectedException

            cv.imshow('Frame', frame)
        except NoHandDetectedException as e:
            cv.putText(frame, 'No hand detected', (10, 450), font, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(frame, f'Prediction: {detector.last_pred}', (10, 500), font, 1, (0, 255, 0), 2, cv.LINE_AA)

            cv.imshow('Frame', frame)

        # print('\r' + str(pipeline), end='')
