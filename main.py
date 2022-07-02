from feature_extraction.mediapipe_landmarks import MediaPipe
from feature_extraction.pipeline import Pipeline
from load_data.by_folder_loader import ByFolderLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from load_data.with_labels_loader import WithLabelsLoader
from preprocessing.all_landmarks import drop_invalid

if __name__ == '__main__':
    pipeline = Pipeline(2)
    pipeline.add_stage()
    pipeline.add_stage(0, 30)
    pipeline.add_stage(0, -30)
    pipeline.add_stage(0, 15)
    pipeline.add_stage(0, -15)
    pipeline.add_stage(60)
    pipeline.add_stage(30)

    loader = WithLabelsLoader(pipeline, 'data/kenyan/images')
    labels = pd.read_csv('data/kenyan/Train.csv')
    loader.create_landmarks(labels)

    # loader = ByFolderLoader(pipeline, 'data/asl_alphabet_train', 1)
    # loader.create_landmarks()

    data = loader.load_landmarks('landmarks_2hands_train.csv')
    X = data.iloc[:, :126]
    y = data['label']
    print(X.head())
    print(len(X))
    print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = KNeighborsClassifier(5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(preds, y_test.to_numpy())
    print(accuracy_score(preds, y_test))
