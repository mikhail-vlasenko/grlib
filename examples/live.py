from src.grlib.exceptions import NoHandDetectedException, HandsAreNotRepresentativeException
from src.grlib.feature_extraction.mediapipe_landmarks import get_landmarks_at_position
from src.grlib.feature_extraction.pipeline import Pipeline
from src.grlib.load_data.by_folder_loader import ByFolderLoader
from src.grlib.filter.false_positive_filter import FalsePositiveFilter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import cv2.cv2 as cv
import numpy as np


if __name__ == '__main__':
    NUM_HANDS = 1
    pipeline = Pipeline(NUM_HANDS)
    pipeline.add_stage(0, 0)

    loader = ByFolderLoader(pipeline, '../data/live')
    loader.create_landmarks()

    data = loader.load_landmarks()
    X = np.array(data.iloc[:, :63])
    y = np.array(data['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = KNeighborsClassifier(5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(preds, y_test)
    print(accuracy_score(preds, y_test))

    camera = cv.VideoCapture(0)

    font = cv.FONT_HERSHEY_SIMPLEX

    fp_filter = FalsePositiveFilter(data, 'cosine')

    run_pipeline = Pipeline(3)
    run_pipeline.add_stage(0, 0)

    while True:
        ret, frame = camera.read()

        if not ret:
            continue

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        try:
            # detect hands on the picture. You can detect more hands than you intend to recognize
            landmarks, handedness = run_pipeline.get_world_landmarks_from_image(frame)
            # get index of the best hand
            index = fp_filter.best_hands_indices(landmarks, handedness, return_hands=NUM_HANDS)[0]
            # select the best hand
            landmarks = get_landmarks_at_position(landmarks, index)
            handedness = handedness[index]

            prediction = model.predict(np.expand_dims(landmarks, axis=0))
            cv.putText(frame, f'Class: {prediction[0]}, {"left" if handedness == 0 else "right"}',
                       (10, 450), font, 1, (0, 255, 0), 2, cv.LINE_AA)

        except HandsAreNotRepresentativeException as e:
            cv.putText(frame, 'No gesture detected', (10, 450), font, 1, (0, 255, 0), 2, cv.LINE_AA)
        except NoHandDetectedException as e:
            cv.putText(frame, 'No hand detected', (10, 450), font, 1, (0, 255, 0), 2, cv.LINE_AA)
        finally:
            cv.imshow('Frame', frame)
