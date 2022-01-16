from feature_extraction.mediapipe_landmarks import MediaPipe
from load_data.default_loader import DefaultLoader
from data_augmentation.common import augment_dataset
from data_augmentation.landmarks_2d import small_rotation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Some notes: add a pipeline, speed up mediapipe


def load_data():
    loader = DefaultLoader('data/asl_kaggle/asl_alphabet_train/asl_alphabet_train')
    # loader.create_landmarks()

    data = loader.load_landmarks()
    X = data.iloc[:, :63]
    X = loader.to_2d(X)
    X_aug = augment_dataset(X, small_rotation)
    X = pd.concat([X, X_aug])
    y = data['label']
    y = pd.concat([y, y])

    loader_left = DefaultLoader('data/asl_left_hand')
    data2 = loader_left.load_landmarks()
    x_test = data2.iloc[:, :63]
    x_test = loader.to_2d(x_test)
    y_test = data2['label']

    # return train_test_split(X, y, test_size=0.2)
    # return x_test[:5000], X[:5000], y_test[:5000], y[:5000]
    return x_test, X, y_test, y


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = KNeighborsClassifier(5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(confusion_matrix(y_test, preds))
    print(accuracy_score(preds, y_test))
