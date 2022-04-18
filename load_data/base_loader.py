from typing import List
import pandas as pd

from exceptions import NoHandDetectedException


class BaseLoader(object):
    """
    Retrieves landmarks from folder with images.
    """

    def __init__(self, path: str, num_hands: int = 2):
        """
        :param path: path to dataset's main folder
        """
        self.num_hands = num_hands
        self.mp = None
        if path[-1] != '/':
            path = path + '/'
        self.path = path

    def create_landmarks_for_image(self, file_path) -> List[float]:
        """
        Processes a single image and retrieves the landmarks of this image.
        :param file_path: - the file path of the file to read
        :return: - the list of landmarks detected by MediaPipe or an empty list if no landmarks were found
        """
        try:
            result = self.mp.get_world_landmarks(file_path).flatten().tolist()
            return result
        except NoHandDetectedException as e:
            print(str(e))
            return list()

    def load_landmarks(self, file='landmarks.csv') -> pd.DataFrame:
        """
        Read landmarks from csv file
        :param file: path, without loader root path
        :return: the dataframe
        """
        return pd.read_csv(self.path + file)
