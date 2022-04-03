import pandas as pd


def is_zeros(data):
    """
    True if all landmarks are 0
    :param data:
    :return:
    """
    # TODO: fix
    return data['0'] == 0


def drop_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all landmarks that only have zeros in them.
    :param df: dataframe with landmarks
    :return: modified df
    """
    return df.drop(df[is_zeros(df)].index)
