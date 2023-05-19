import numpy as np
import pandas as pd
from os.path import join

from ..feature_extraction.pipeline import Pipeline
from ..load_data.base_loader import BaseLoader


class WithLabelsLoader(BaseLoader):
    """
    Retrieves landmarks from folder with images.
    """

    def __init__(self, pipeline: Pipeline, path: str, verbose: bool = True):
        """
        :param path: path to dataset's main folder
        :param num_hands: the number of hands to detect
        """
        super().__init__(pipeline, path, verbose)

    def create_landmarks(self, labels: pd.DataFrame, output_file='landmarks.csv'):
        """
        Processes images of gestures and saves results to csv.
        Images are labelled according to provided *labels*
        If no hand is found, all 63 landmarks are set to 0
        :param labels: a dataframe with additional image path (no dataset root) in *path* column and *label* column.
        If no *label* column is provided, landmarks are computed, but no labels are assigned.
        :param output_file: the file path of the file to write to
        :return: None
        """
        files = join(self.path, labels['path'] + '.jpg')

        results = []
        handedness_results = []
        for i, f in enumerate(files):
            landmarks, handedness = self.create_landmarks_for_image(f)
            results.append(landmarks)
            handedness_results.append(handedness)

        # Replace with 0s to keep the correct order with respect to the labels file
        results = [res if len(res) > 0 else np.zeros(self.pipeline.num_hands * 63) for res in results]

        if 'label' in labels.columns:
            label_column = labels['label']
        else:
            label_column = None
        df = BaseLoader.make_df_with_handedness(np.array(results), np.array(handedness_results), label_column)
        df.to_csv(self.path + output_file, index=False)
