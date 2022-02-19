import os
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from typing import List

import numpy as np
import pandas as pd
import cv2 as cv

from exceptions import NoHandDetectedException
from feature_extraction.mediapipe_landmarks import MediaPipe


class DefaultLoader:
    """
    Retrieves landmarks from folder with images.
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
        Processes images of gestures and saves results to csv.
        Images are labelled with their folder's name.
        takes a while
        :param output_file: the file path of the file to write to
        :return: None
        """
        self.mp = MediaPipe()
        data = []
        data_labels = [folder for folder in os.listdir(self.path) if os.path.isdir(self.path + folder)]

        # go through all folders found in this directory
        for i, folder in enumerate(data_labels):
            curr_path = self.path + folder + '/'
            print(f'Processing {curr_path}')
            files = [curr_path + file for file in os.listdir(curr_path)]

            # use all available processors for faster computation
            results = self.thread_pool.map(self.create_landmarks_for_image, files)
            results = [result for result in results if len(result) > 0]
            # append label
            for result in results:
                result.append(folder)

            data.extend(results)
        self.mp.close()

        data = np.array(data)
        df = pd.DataFrame(data)
        # rename label column, others stay as numbers
        df = df.rename(columns={len(df.columns)-1: 'label'})
        df.to_csv(self.path + output_file, index=False)

    def create_landmarks_with_labels(self, labels: pd.DataFrame, output_file='landmarks.csv', threading=True):
        """
        Processes images of gestures and saves results to csv.
        Images are labelled according to provided *labels*
        If no hand is found, all 63 landmarks are set to 0
        :param labels: a dataframe with additional image path (no dataset root) in *path* column and *label* column.
        If no *labels* column is provided, landmarks are computed, but no labels are assigned.
        :param output_file: the file path of the file to write to
        :param threading: use threading to speed up (may lead to scheduler error)
        :return: None
        """
        files = self.path + labels['path']

        self.mp = MediaPipe()
        if threading:
            results = self.thread_pool.map(self.create_landmarks_for_image, files)
        else:
            results = []
            for f in files:
                results.append(self.create_landmarks_for_image(f))
        self.mp.close()

        results = [res if len(res) > 0 else np.zeros(63 * 2) for res in results]

        df = pd.DataFrame(np.array(results))
        if 'label' in labels.columns:
            df['label'] = labels['label']
        df.to_csv(self.path + output_file, index=False)

    def create_landmarks_for_image(self, file_path) -> List[object]:
        """
        Processes a single image and retrieves the landmarks of this image. Used by the threads.
        :param file_path: - the file path of the file to read
        :return: - the list of 63 landmarks detected by MediaPipe or an empty list if no landmarks were found
        """
        try:
            result = self.mp.get_world_landmarks(file_path).flatten().tolist()
            return result
        except NoHandDetectedException as e:
            cv.imwrite('data/kenyan/out/' + os.path.basename(file_path), cv.imread(file_path))
            # Print it like this to avoid buffer issues in multi-threaded code
            print(str(e) + '\n', end='')
            return list()

    def load_landmarks(self, file='landmarks.csv') -> pd.DataFrame:
        """
        Read landmarks from csv file
        :param file: path, without loader root path
        :return: the dataframe
        """
        return pd.read_csv(self.path + file)
