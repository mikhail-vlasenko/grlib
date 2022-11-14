from typing import List
import pandas as pd
import numpy as np
from ..exceptions import NoHandDetectedException
from ..feature_extraction.pipeline import Pipeline


class BaseLoader(object):
    """
    Retrieves landmarks from folder with images.
    """

    def __init__(self, pipeline: Pipeline, path: str, verbose: bool = True):
        """
        :param pipeline: the pipeline the loader should use to detect landmarks
        :param path: path to dataset's main folder
        :param verbose: whether to display the pipeline after each step
        """
        self.verbose = verbose
        self.pipeline = pipeline
        self.mp = None
        if path[-1] != '/':
            path = path + '/'
        self.path = path

    def create_landmarks_for_image(self, file_path, world_landmarks=True) -> np.ndarray:
        """
        Processes a single image and retrieves the landmarks of this image.
        :param file_path: the file path of the file to read
        :param world_landmarks: whether to return world landmarks (centered on the hand) or
        landmarks relative to the image borders.
        :return: the list of landmarks detected by MediaPipe or an empty list if no landmarks were found
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
            return np.array([])

    def load_landmarks(self, file='landmarks.csv') -> pd.DataFrame:
        """
        Read landmarks from csv file
        :param file: path, without loader root path
        :return: the dataframe
        """
        return pd.read_csv(self.path + file)
