import pandas as pd


def to_2d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that store depth coordinate
    :param df: the dataframe
    :return: dataframe without depth columns
    """
    drop_columns = [str(x) for x in range(2, len(df.columns), 3)]
    return df.drop(columns=drop_columns)
