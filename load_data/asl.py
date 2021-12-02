import os
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from typing import List

import numpy as np
import pandas as pd

from exceptions import NoHandDetectedException
from feature_extraction.mediapipe_landmarks import MediaPipe


class ASLLoader:
    """
    retrieves landmarks from unpacked dataset
    https://www.kaggle.com/vaishnaviasonawane/indian-sign-language-dataset
    """
    def __init__(self, path):
        """
        :param path: path to dataset's main folder
        """
        if path[-1] != '/':
            path = path + '/'
        self.path = path
        self.thread_pool = ThreadPool(cpu_count())

    def create_digit_landmarks(self, output_file='digit_landmarks.csv'):
        """
        processes images of digit gestures and saves results to csv
        takes a while
        :param output_file:
        :return:
        """
        self.mp = MediaPipe()
        data = []
        data_labels = [folder for folder in os.listdir(self.path) if os.path.isdir(self.path + folder)]

        for i, folder in enumerate(data_labels):
            curr_path = self.path + folder + '/'
            print(f'Processing {curr_path}')

            files = [curr_path + file for file in os.listdir(curr_path)]

            results = self.thread_pool.map(self.create_digit_landmark, files)
            results = [result for result in results if len(result) > 0]
            for result in results:
                result.append(i)

            data.extend(results)
        self.mp.close()

        data = np.array(data)
        df = pd.DataFrame(data)
        # rename label column, others stay as numbers
        df = df.rename(columns={len(df.columns)-1: "label"})
        df.to_csv(self.path + output_file, index=False)

    def create_digit_landmark(self, file_path) -> List[object]:
        """
        Processes a single image and retrieves the landmarks of this image. Used by the threads.
        :param file_path: - the file path of the file to read
        :return: - the list of landmarks detected by MediaPipe or an empty list if no landmarks were found
        """
        try:
            result = self.mp.get_world_landmarks(file_path).flatten().tolist()
            return result
        except NoHandDetectedException as e:
            print(e)
            return list()

    def load_digit_landmarks(self, file='digit_landmarks.csv') -> pd.DataFrame:
        return pd.read_csv(self.path + file)
