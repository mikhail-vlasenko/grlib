import os
import numpy as np
import pandas as pd

from feature_extraction.mediapipe_landmarks import MediaPipe
from load_data.base_loader import BaseLoader


class ByFolderLoader(BaseLoader):
    """
    Retrieves landmarks from folder with images.
    """
    def __init__(self, path):
        """
        :param path: path to dataset's main folder
        """
        super().__init__(path)

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

        for i, folder in enumerate(data_labels):
            curr_path = self.path + folder + '/'
            print(f'Processing {curr_path}')

            files = [curr_path + file for file in os.listdir(curr_path)]

            results = []
            for file in files:
                results.append(self.create_landmarks_for_image(file))

            # Remove the instances where no hand was detected
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
