from os import listdir

import numpy as np

from feature_extraction.mediapipe_landmarks import MediaPipe


class IslLoader:
    def __init__(self, path):
        if path[-1] != '/':
            path = path + '/'
        self.path = path

    def create_digit_landmarks(self, output_file='digit_landmarks'):
        """
        processes images of digit gestures and saves np array to binary file
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
                data.append(mp.get_world_landmarks(curr_path + file))
        data = np.array(data)
        np.save(self.path + output_file, data)

    def load_digit_landmarks(self, file='digit_landmarks.npy'):
        return np.load(self.path + file)
