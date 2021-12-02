import os
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from typing import List

import numpy as np
import pandas as pd

from exceptions import NoHandDetectedException
from feature_extraction.mediapipe_landmarks import MediaPipe


class DefaultLoader:
    """
    retrieves landmarks from folder with images
    """
    def __init__(self, path):
        """
        :param path: path to dataset's main folder
        """
        self.mp = None
        if path[-1] != '/':
            path = path + '/'
        self.path = path
        self.thread_pool = ThreadPool(cpu_count())

    def create_landmarks(self, output_file='landmarks.csv'):
        """
        processes images of gestures and saves results to csv
        images are labelled with their folder's name
        takes a while
        :param output_file: the file path of the file to write to
        :return: None
        """
        self.mp = MediaPipe()
        data = []
        data_labels = [folder for folder in os.listdir(self.path) if os.path.isdir(self.path + folder)]

        for i, folder in enumerate(data_labels):
            curr_path = self.path + folder + '/'
            print(f'Processing {curr_path}')

            files = [curr_path + file for file in os.listdir(curr_path)]

            results = self.thread_pool.map(self.create_landmarks_for_image, files)
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

    def create_landmarks_for_image(self, file_path) -> List[object]:
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

    def load_landmarks(self, file='landmarks.csv') -> pd.DataFrame:
        return pd.read_csv(self.path + file)
