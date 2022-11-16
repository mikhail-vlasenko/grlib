from typing import List
import pandas as pd
import numpy as np
from ..exceptions import NoHandDetectedException
from ..feature_extraction.pipeline import Pipeline


class BaseLoader(object):
    """
    Retrieves landmarks from folder with images.
    """

    def __init__(self, pipeline: Pipeline, path: str,
                 use_handedness: bool = True, verbose: bool = True):
        """
        :param pipeline: the pipeline the loader should use to detect landmarks
        :param path: path to dataset's main folder
        :param use_handedness: include info whether the hand is right or left
        :param verbose: whether to display the pipeline after each step
        """
        self.verbose = verbose
        self.pipeline = pipeline
        if path[-1] != '/':
            path = path + '/'
        self.path = path
        self.use_handedness = use_handedness

    def create_landmarks_for_image(self, file_path, world_landmarks=True) -> (np.ndarray, np.ndarray):
        """
        Processes a single image and retrieves the landmarks of this image.
        :param file_path: the file path of the file to read
        :param world_landmarks: whether to return world landmarks (centered on the hand) or
        landmarks relative to the image borders.
        :return: (the list of landmarks detected by MediaPipe
        or an empty list if no landmarks were found,
        list of handedness labels (left/right))
        """
        try:
            if world_landmarks:
                result = self.pipeline.get_world_landmarks_from_path(file_path)
            else:
                result = self.pipeline.get_landmarks_from_path(file_path)
            self.pipeline.optimize()
            if self.verbose:
                print('\r' + str(self.pipeline), end='')
            return result
        except NoHandDetectedException as e:
            # print(str(e))
            if self.verbose:
                print('\r' + str(self.pipeline), end='')
            return np.array([]), np.array([])

    def load_landmarks(self, file='landmarks.csv') -> pd.DataFrame:
        """
        Read landmarks from csv file
        :param file: path, without loader root path
        :return: the dataframe
        """
        # todo: return in separate arrays
        return pd.read_csv(self.path + file)
