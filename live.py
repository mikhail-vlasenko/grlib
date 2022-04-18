from feature_extraction.mediapipe_landmarks import MediaPipe
from load_data.by_folder_loader import ByFolderLoader
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    loader = ByFolderLoader('out')
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

        landmarks = loader.mp.process_from_image(frame).multi_hand_world_landmarks

        if landmarks is None:
            cv.putText(frame, 'No hand detected', (10, 450), font, 1, (0, 255, 0), 2, cv.LINE_AA)

            cv.imshow('Frame', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            continue

        hand = landmarks[0].landmark

        point_array = []
        for point in hand:
            point_array.extend([point.x, point.y, point.z])

        prediction = model.predict(np.array([point_array]))

        cv.putText(frame, f'Class: {prediction[0]}', (10, 450), font, 1, (0, 255, 0), 2, cv.LINE_AA)

        cv.imshow('Frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
