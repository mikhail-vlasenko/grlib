from feature_extraction.mediapipe_landmarks import MediaPipe
from load_data.default_loader import DefaultLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Some notes: add a pipeline, speed up mediapipe

if __name__ == '__main__':
    loader = DefaultLoader('data/asl_alphabet_train/')
    loader.create_landmarks()

    data = loader.load_landmarks()
    X = data.iloc[:, :63]
    y = data['label']
    print(X.head())
    print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = KNeighborsClassifier(5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(preds, y_test.to_numpy())
    print(accuracy_score(preds, y_test))