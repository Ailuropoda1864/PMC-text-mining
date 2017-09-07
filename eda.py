import numpy as np
import pandas as pd
import re
from pandas.core.dtypes.common import (
    is_numeric_dtype, is_datetime64_dtype, is_bool_dtype
)
from pandas.core.indexes.datetimes import DatetimeIndex


def eda(dataframe, head=True, info=True, describe=True, duplicated=True,
        dup_kwd={}):
    """
    exploratory data analysis
    :param dataframe: a pandas DataFrame
    :param head: boolean; if True, the first 5 rows of dataframe is shown
    :param info: boolean; if True, dataframe.info() and nulls are shown
    :param describe: boolean; if True, descriptions of the columns (grouped by
                     numeric, datetime, and other) are shown
    :param duplicated: boolean; if True, info on duplicated rows are shown
    :param dup_kwd: keyword arguments for find_duplicated
    :return: None
    """
    assert isinstance(dataframe, pd.DataFrame), \
        "pandas DataFrame is required; got {} instead".format(type(dataframe))

    if head:
        print('Head of the dataframe:\n\n{}\n\n'.format(dataframe.head()))

    # shape, index, columns, nulls, dtypes
    if info:
        dataframe.info()
        print('\n')
        show_null(dataframe)
        print('\n')

    if describe:
        describe_by_type(dataframe)

    # find duplicates
    if duplicated:
        find_duplicate(dataframe, **dup_kwd)


def describe_by_type(dataframe):
    """
    prints descriptions of the columns (grouped by numeric, datetime, boolean,
    and others) and DatetimeIndex (if any)
    :param dataframe: a pandas DataFrame
    :return: None
    """
    boolean, numeric, datetime, other = False, False, False, False
    for column in dataframe.columns:
        if is_bool_dtype(dataframe[column]):
            boolean = True
        elif is_numeric_dtype(dataframe[column]):
            numeric = True
        elif is_datetime64_dtype(dataframe[column]):
            datetime = True
        else:
            other = True

    # describe datetime columns and DatetimeIndex (if any)
    if isinstance(dataframe.index, DatetimeIndex):
        print(pd.Series(dataframe.index).describe())
        print('\n')

    if datetime:
        print(dataframe.describe(include=['datetime']))
        print('\n')

    # describe numeric columns (if any)
    if numeric:
        print(dataframe.describe())
        print('\n')

    # describe boolean columns (if any)
    if boolean:
        print(dataframe.describe(include=[np.bool]))
        print('\n')

    # describe other columns (if any)
    if other:
        print(dataframe.describe(exclude=[np.number, np.datetime64, np.bool]))
        print('\n')


def show_null(dataframe):
    """
    prints the number and percentage of null values in each column
    :param dataframe: a pandas DataFrame
    :return: None
    """
    if dataframe.isnull().sum().sum() == 0:
        print('No null in the dataframe.')
    else:
        print('Number of nulls in each column:\n{}\n'.format(
            dataframe.isnull().sum()
        ))
        print('Percentage of nulls in each column:\n{}\n'.format(
            dataframe.isnull().sum() / len(dataframe)
        ))


def find_duplicate(dataframe, show=True, sort=False):
    """
    prints out information on duplicate rows
    :param dataframe: a pandas DataFrame
    :param show: boolean; if True, the duplicated rows (if any) are shown
    :param sort: boolean; if True, the duplicated rows are sorted by each column
                 of the dataframe
    """
    n_duplicates = dataframe.duplicated().sum()
    print('Number of duplicated rows: {}'.format(n_duplicates))
    if show and n_duplicates > 0:
        print()
        duplicated_df = dataframe[dataframe.duplicated(keep=False)]
        if sort:
            print(duplicated_df.sort_values(list(duplicated_df.columns)))
        else:
            print(duplicated_df)


def category_counts(dataframe, max_nunique=20, numeric=False, datetime=False):
    """
    prints value counts for each (categorical) column
    :param dataframe: a pandas DataFrame
    :param max_nunique: the max number of unique values a column can have for
                        its value counts to be printed; no limit is set if None
    :param numeric: boolean; if True, value counts for numeric data are also
                    printed
    :param datetime: boolean; if True, value counts for datetime data are also
                     printed
    :return: None
    """
    for column in dataframe.columns:
        col = dataframe[column]
        if is_bool_dtype(col):
            print(col.value_counts())
            print('\n')
            break
        if not any([
            max_nunique is not None and col.nunique() > max_nunique,
            not numeric and is_numeric_dtype(col),
            not datetime and is_datetime64_dtype(col)
        ]):
            print(col.value_counts())
            print('\n')



