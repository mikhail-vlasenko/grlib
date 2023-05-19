# Attempt to import grlib as a dependency, if that fails, try to assume it is a local project
try:
    from grlib.feature_extraction.pipeline import Pipeline
    from grlib.load_data.by_folder_loader import ByFolderLoader
except ImportError as ex:
    from src.grlib.feature_extraction.pipeline import Pipeline
    from src.grlib.load_data.by_folder_loader import ByFolderLoader

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

if __name__ == '__main__':
    pipeline = Pipeline(num_hands=2)
    pipeline.add_stage(0, 0)
    pipeline.add_stage(30, 0)
    pipeline.add_stage(60, 0)
    pipeline.add_stage(15, -15)
    pipeline.add_stage(15, 15)
    pipeline.add_stage(0, 30)
    pipeline.add_stage(0, -30)

    loader = ByFolderLoader(pipeline, '../data/ASL_Dataset/Train')
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
