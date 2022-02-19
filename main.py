from feature_extraction.mediapipe_landmarks import MediaPipe
from load_data.default_loader import DefaultLoader
from data_augmentation.common import augment_dataset
from data_augmentation.landmarks_2d import small_rotation, scaling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocessing.spacial_landmarks import to_2d
from preprocessing.all_landmarks import drop_invalid
from sklearn.model_selection import GridSearchCV


def load_data():
    labels = pd.read_csv('data/kenyan/Train.csv')
    labels = labels.rename(columns={'img_IDS': 'path', 'Label': 'label'})
    labels['path'] = labels['path'].apply(lambda s: s + '.jpg')
    loader = DefaultLoader('data/kenyan')
    # loader.create_landmarks_with_labels(labels, threading=False)

    data = loader.load_landmarks('landmarks_2hands_train.csv')
    data = drop_invalid(data)
    X = data.iloc[:, :-1]
    y = data['label']

    return X, y


def load_test():
    labels = pd.read_csv('data/kenyan/Test.csv')
    labels = labels.rename(columns={'img_IDS': 'path'})
    labels['path'] = labels['path'].apply(lambda s: s + '.jpg')
    loader = DefaultLoader('data/kenyan')
    # loader.create_landmarks_with_labels(labels, threading=False)

    data = loader.load_landmarks('landmarks_2hands_test.csv')
    return data


def find_best_model(models, params, X, y):
    best_model = None
    best_score = 0
    for model, param in zip(models, params):
        grid = GridSearchCV(model, param, scoring='accuracy')

        grid.fit(X, y)

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_

        print(grid.best_estimator_)
        print(grid.best_score_)
        print()

    return best_model

if __name__ == '__main__':
    X, y = load_data()

    params = [
        {'strategy': ['stratified', 'most_frequent', 'uniform']},
        {'n_neighbors': [3, 5, 7, 11, 15], 'weights': ['uniform', 'distance']},
        {'C': [1, 3, 5, 10, 20], 'degree': [1, 2, 3]},
        {'C': [40, 80, 100, 150, 300]},
        {'max_depth': [None, 2, 5, 10, 20, 30], 'criterion': ['gini', 'entropy']}
    ]

    models = [DummyClassifier(), KNeighborsClassifier(), SVC(), LogisticRegression(), DecisionTreeClassifier()]

    # best_model = find_best_model(models, params, X, y)
    best_model = SVC(C=10, degree=1)
    best_model.fit(X, y)

    X_test = load_test()

    preds = best_model.predict(X_test)
    print(preds)

    classes = ['Church', 'Enough/Satisfied', 'Friend', 'Love', 'Me', 'Mosque', 'Seat', 'Temple', 'You']

    labels = pd.read_csv('data/kenyan/Test.csv').to_numpy()

    result = np.full((labels.shape[0], len(classes)), 0.2)
    for i, pred in enumerate(preds):
        inx = classes.index(pred)
        result[i][inx] = 0.8

    result = pd.DataFrame(result, columns=classes)
    result.insert(0, 'img_IDS', labels, True)
    result.to_csv('predictions.csv', index=False)

    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    #
    # model = KNeighborsClassifier(5)
    # model.fit(X_train, y_train)
    #
    # preds = model.predict(X_test)
    # print(confusion_matrix(y_test, preds))
    # print(accuracy_score(preds, y_test))
