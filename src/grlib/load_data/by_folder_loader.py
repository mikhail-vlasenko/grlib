import os
import numpy as np
import pandas as pd

from ..feature_extraction.pipeline import Pipeline
from ..load_data.base_loader import BaseLoader


class ByFolderLoader(BaseLoader):
    """
    Retrieves landmarks from folder with images.
    """
    def __init__(self, pipeline: Pipeline, path: str, verbose: bool = True):
        """
        :param path: path to dataset's main folder
        :param num_hands: the number of hands to detect
        """
        super().__init__(pipeline, path, verbose)

    def create_landmarks(self, output_file='landmarks.csv'):
        """
        Processes images of gestures and saves results to csv.
        Images are labelled with their folder's name.
        takes a while
        :param output_file: the file path of the file to write to
        :return: None
        """

        # Potential speed-up: create mp instance per thread - this should be thread safe
        data = []
        data_labels = [folder for folder in os.listdir(self.path) if os.path.isdir(self.path + folder)]

        for i, folder in enumerate(data_labels):
            curr_path = self.path + folder + '/'
            print(f'Processing {curr_path}')

            files = [curr_path + file for file in os.listdir(curr_path)]

            results = []
            for file_idx, file in enumerate(files):
                results.append(self.create_landmarks_for_image(file))

            if self.verbose:
                print()

            # Remove the instances where no hand was detected
            results = [result for result in results if len(result) > 0]

            for result in results:
                result.append(i)

            data.extend(results)

        data = np.array(data)
        df = pd.DataFrame(data)

        # rename label column, others stay as numbers
        df = df.rename(columns={len(df.columns)-1: "label"})
        df.to_csv(self.path + output_file, index=False)
