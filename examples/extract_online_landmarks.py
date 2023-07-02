try:
    from grlib.exceptions import NoHandDetectedException, HandsAreNotRepresentativeException
    from grlib.feature_extraction.mediapipe_landmarks import get_landmarks_at_position
    from grlib.feature_extraction.pipeline import Pipeline
    from grlib.load_data.by_folder_loader import ByFolderLoader
    from grlib.filter.false_positive_filter import FalsePositiveFilter
except ImportError as ex:
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


pipeline = Pipeline(1)
pipeline.add_stage(0, 0)
loader = ByFolderLoader(pipeline, '../data/dynamic_dataset_online')
loader.create_landmarks(save_positions=True)
