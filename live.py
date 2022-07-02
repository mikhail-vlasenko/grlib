from exceptions import NoHandDetectedException
from feature_extraction.mediapipe_landmarks import MediaPipe
from feature_extraction.pipeline import Pipeline
from load_data.by_folder_loader import ByFolderLoader
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    pipeline = Pipeline(1)
    pipeline.add_stage(0, 0)
    pipeline.add_stage(30, 0)
    pipeline.add_stage(60, 0)
    pipeline.add_stage(30, -15)
    pipeline.add_stage(30, 15)

    loader = ByFolderLoader(pipeline, 'out')
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

    loader.mp = MediaPipe()

    font = cv.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = camera.read()

        if not ret:
            continue

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        try:
            landmarks = pipeline.get_world_landmarks_from_image(frame).flatten().tolist()
            pipeline.optimize()

            prediction = model.predict(np.array([landmarks]))

            cv.putText(frame, f'Class: {prediction[0]}', (10, 450), font, 1, (0, 255, 0), 2, cv.LINE_AA)

            cv.imshow('Frame', frame)
        except NoHandDetectedException as e:
            cv.putText(frame, 'No hand detected', (10, 450), font, 1, (0, 255, 0), 2, cv.LINE_AA)

            cv.imshow('Frame', frame)

        print('\r' + str(pipeline), end='')
