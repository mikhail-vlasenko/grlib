from os import listdir

import numpy as np
import pandas as pd

from feature_extraction.mediapipe_landmarks import MediaPipe


class DefaultLoader:
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

    def create_digit_landmarks(self, output_file='digit_landmarks.csv'):
        """
        processes images of digit gestures and saves results to csv
        takes a while
        :param output_file:
        :return:
        """
        mp = MediaPipe()
        data = []
        for i in range(1, 10):
            curr_path = self.path + str(i) + '/'
            print('processing ' + curr_path)
            for file in listdir(curr_path):
                # calculate landmarks
                landmarks = mp.get_world_landmarks(curr_path + file).flatten().tolist()
                landmarks.append(i)  # add label
                data.append(landmarks)
        data = np.array(data)
        df = pd.DataFrame(data)
        # rename label column, others stay as numbers
        df = df.rename(columns={len(df.columns)-1: "label"})
        df.to_csv(self.path + output_file, index=False)

    def load_digit_landmarks(self, file='digit_landmarks.csv') -> pd.DataFrame:
        return pd.read_csv(self.path + file)
