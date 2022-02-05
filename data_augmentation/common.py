import pandas as pd
from typing import Callable


def augment_dataset(data: pd.DataFrame, augmentation: Callable) -> pd.DataFrame:
    """
    computes augmented data points
    :return: pd DataFrame with augmented landmarks
    """
    return data.apply(augmentation, axis=1, result_type='broadcast')
