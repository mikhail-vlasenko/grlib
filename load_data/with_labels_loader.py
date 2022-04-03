import os
import numpy as np
import pandas as pd

from feature_extraction.mediapipe_landmarks import MediaPipe
from load_data.base_loader import BaseLoader


class WithLabelsLoader(BaseLoader):
    """
    Retrieves landmarks from folder with images.
    """

    def __init__(self, path: str, num_hands: int = 2):
        """
        :param path: path to dataset's main folder
        """
        super().__init__(path, num_hands)

    def create_landmarks(self, labels: pd.DataFrame, output_file='landmarks.csv'):
        """
        Processes images of gestures and saves results to csv.
        Images are labelled according to provided *labels*
        If no hand is found, all 63 landmarks are set to 0
        :param labels: a dataframe with additional image path (no dataset root) in *path* column and *label* column.
        If no *labels* column is provided, landmarks are computed, but no labels are assigned.
        :param output_file: the file path of the file to write to
        :return: None
        """
        files = self.path + labels['path']

        self.mp = MediaPipe()

        results = []
        for f in files:
            results.append(self.create_landmarks_for_image(f))

        # Replace with 0s to keep the correct order with respect to the labels file
        results = [res if len(res) > 0 else np.zeros(63) for res in results]

        self.mp.close()

        df = pd.DataFrame(np.array(results))
        if 'label' in labels.columns:
            df['label'] = labels['label']
        df.to_csv(self.path + output_file, index=False)
