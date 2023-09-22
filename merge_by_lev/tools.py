from typing import List, Tuple

import pandas as pd
from pandas.core.frame import DataFrame


def check_empty_df(
    dfs: List[DataFrame], names: List[str], num_cols: str = 2
) -> Tuple[List[DataFrame], List[str]]:
    """Check if the `DataFrame` is empty or not

    Args:
        dfs (List[`DataFrame`]): List of dataframes to iterate over
        names (List[`str`]): List of `DataFrame` names

    Returns:
        Tuple[List[DataFrame], List[str]]: Verified dataframes and names
    """
    new_dfs = []
    new_names = []
    for i, df in enumerate(dfs):
        if len(df.columns) > num_cols:
            if len(df) > num_cols:
                new_dfs.append(df)
                new_names.append(names[i])

    return new_dfs, new_names
