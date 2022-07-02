import pandas as pd


def is_zeros(data):
    """
    True if all landmarks are 0
    :param data: the data to check for empty landmarks
    :return: the indexes where all columns are 0s (meaning no landmarks)
    """
    return (data.iloc[:, :-2] == 0).all(axis=1)


def drop_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all landmarks that only have zeros in them.
    :param df: dataframe with landmarks
    :return: modified df
    """
    return df.drop(df[is_zeros(df)].index)
