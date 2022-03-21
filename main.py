from feature_extraction.mediapipe_landmarks import MediaPipe
from load_data.default_loader import DefaultLoader
from data_augmentation.common import augment_dataset
from data_augmentation.landmarks_2d import small_rotation, scaling
from preprocessing.all_landmarks import is_zeros
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


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
    classes = ['Church', 'Enough/Satisfied', 'Friend', 'Love', 'Me', 'Mosque', 'Seat', 'Temple', 'You']

    X, y = load_data()

    params = [
        {'strategy': ['stratified', 'most_frequent', 'uniform']},
        {'n_neighbors': [3, 5, 7, 11, 15], 'weights': ['uniform', 'distance']},
        {'C': [5, 10, 20, 40, 60, 80, 100, 150], 'degree': [1, 2, 3]},
        {'C': [40, 80, 100, 150, 300]},
        {'max_depth': [None, 2, 5, 10, 20, 30], 'criterion': ['gini', 'entropy']}
    ]

    models = [DummyClassifier(), KNeighborsClassifier(), SVC(), LogisticRegression(), DecisionTreeClassifier()]

    y_nn = np.zeros((y.shape[0], len(classes)))
    for i in range(y.shape[0]):
        inx = classes.index(y.iloc[i])
        y_nn[i][inx] = 1

    checkpoint_callback = ModelCheckpoint(filepath='weights_{val_categorical_accuracy:.3f}.h5',
                               monitor='val_categorical_accuracy',
                               mode='max',
                               save_best_only=True)

    nn = Sequential()
    nn.add(Conv1D(128, (12, ), activation='relu', input_shape=(X.shape[1], 1)))
    nn.add(Flatten())
    nn.add(Dense(256, activation='relu'))
    nn.add(Dropout(0.4))
    nn.add(Dense(64, activation='relu'))
    nn.add(Dropout(0.2))
    nn.add(Dense(16, activation='relu'))
    nn.add(Dropout(0.2))
    nn.add(Dense(y_nn.shape[1], activation='sigmoid'))

    nn.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['categorical_accuracy'])

    # nn.fit(X, y_nn, batch_size=32, epochs=60,
    #        validation_split=0.2,
    #        callbacks=[checkpoint_callback])


    # best_model = find_best_model(models, params, X, y)
    # best_model = SVC(C=10, degree=1)
    # best_model.fit(X, y)

    X_test = load_test()
    model = load_model('weights_0.916.h5')
    preds = model.predict(X_test)


    labels = pd.read_csv('data/kenyan/Test.csv').to_numpy()
    preds_resnet = pd.read_csv('data/kenyan/eff_long_subm.csv').to_numpy()
    landmarks_test = pd.read_csv('data/kenyan/landmarks_2hands_test.csv').to_numpy()

    result = np.full((labels.shape[0], len(classes)), 0.01)
    for i, pred in enumerate(preds):
        if landmarks_test[i][0] != 0:
            inx = np.argmax(pred)
            result[i][inx] = 0.99
        else:
            max_inx = np.argmax(preds_resnet[i][1:])
            result[i][max_inx] = 0.99

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
