import os
import platform
import re
import sys
import warnings
from functools import lru_cache
from io import TextIOWrapper
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from IPython.display import clear_output
from numpy import ndarray
from pandas.core.frame import DataFrame
from tabulate import tabulate

from merge_by_lev.tools import check_empty_df


def progressbar(
    it: range, prefix: str = "", size: int = 40, out: TextIOWrapper = sys.stdout
) -> None:  # type: ignore
    """
    Displays a progress bar in the console.

    Parameters
    ----------
    it : `range`
        The iterable to track progress on.
    prefix : `str`, optional
        Prefix text for the progress bar. Default is an empty string.
    size : `int`, optional
        Width of the progress bar. Default is 40 characters.
    out : `TextIOWrapper`, optional
        Output stream where the progress bar will be displayed. Default is `sys.stdout`.
    """
    count = len(it)

    def show(j: int) -> None:
        x = int(size * j / count)
        print(f"{prefix}[{'#' * x}{'.' * (size - x)}] {j}/{count}", end="\r", file=out, flush=True)

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)


def clear_console() -> None:
    """
    Clears the console screen.

    This function uses platform-specific commands to clear the console screen.
    It supports Unix-like systems and Windows.
    """
    command = "clear" if platform.system() != "Windows" else "cls"
    os.system(command)


def check_cols_to_match(dict_dfs: dict[str, DataFrame], df_names: List[str]) -> None:
    """
    Checks if the dataframes in the dictionary have matching columns.

    Parameters
    ----------
    dict_dfs : `dict`
        Dictionary mapping dataframe names to DataFrames.
    df_names : `List[str]`
        List of dataframe names to check.

    Returns
    -------
    None

    Prints the result for each dataframe indicating whether it has matching columns.
    """
    cols_set = set().union(*(set(df.columns) for name in df_names for df in dict_dfs.values()))

    for name in df_names:
        col_set = set(dict_dfs[name].columns)
        if col_set == cols_set:
            print(f"{name} -> Passed")
        else:
            print(f"{name} -> Not Passed")
            print(f"Columns do not match -> {cols_set - col_set}")
            print(f"Original columns -> {col_set}\n")


def rename_cols(df: DataFrame) -> DataFrame:
    """
    Renames columns in the dataframe after merging to consolidate similar columns.

    Parameters
    ----------
    df : `DataFrame`
        The dataframe to operate on.

    Returns
    -------
    `DataFrame`
        The dataframe with consolidated and renamed columns.
    """
    cols = set(col.replace("_x", "").replace("_y", "") for col in df.columns)

    rename_dict = {f"{col}_x": col for col in cols if f"{col}_x" in df.columns}
    drop_cols = [f"{col}_y" for col in cols if f"{col}_y" in df.columns]

    df.rename(columns=rename_dict, inplace=True)
    df.drop(columns=drop_cols, inplace=True)

    return df


def clean_names(x: str, pattern: str = r"[a-zA-Zñáéíóú]*") -> str:
    """
    Cleans the input string to extract only alphabetic characters and Spanish accented vowels.

    Parameters
    ----------
    x : `str`
        The input string to clean.
    pattern : `regex`, optional
        Regex pattern to apply for cleaning. Defaults to extracting alphabetic characters and accented vowels.

    Returns
    -------
    `str`
        The cleaned string containing only the specified characters.
    """
    return "_".join(re.findall(pattern, x))


def lev_dist(a: str, b: str) -> int:
    """
    Calculates the Levenshtein distance between two input strings.

    Parameters
    ----------
    a : `str`
        The first string to compare.
    b : `str`
        The second string to compare.

    Returns
    -------
    `int`
        The Levenshtein distance between the two strings.
    """

    @lru_cache(None)
    def min_dist(s1: int, s2: int) -> int:
        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(min_dist(s1, s2 + 1), min_dist(s1 + 1, s2), min_dist(s1 + 1, s2 + 1))

    return min_dist(0, 0)


def cal_cols_similarity(col_list: List[str]) -> ndarray:
    """
    Calculates the pairwise Levenshtein distance between column names.

    Parameters
    ----------
    col_list : `List[str]`
        List of column names to compare.

    Returns
    -------
    `np.array`
        Matrix containing the pairwise Levenshtein distances between column names.
    """
    n = len(col_list)
    mtx = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            mtx[i, j] = lev_dist(col_list[i], col_list[j])
            mtx[j, i] = mtx[i, j]
    return mtx


def create_table_tabular(df1: DataFrame, df2: DataFrame) -> List[List[str]]:
    """
    Creates a tabular representation of column names from two dataframes.

    Parameters
    ----------
    df1 : `DataFrame`
        First dataframe.
    df2 : `DataFrame`
        Second dataframe.

    Returns
    -------
    `List[List[str]]`
        List of rows containing the column names of both dataframes.
    """
    table = []
    col_names_df1 = df1.columns.tolist()
    col_names_df2 = df2.columns.tolist()
    max_length = max(len(col_names_df1), len(col_names_df2))
    for i in range(max_length):
        col1 = col_names_df1[i] if i < len(col_names_df1) else ""
        col2 = col_names_df2[i] if i < len(col_names_df2) else ""
        table.append([col1, col2])
    return table


def rename_cols_dict(df_name: str, df: DataFrame, cols: List[str]) -> DataFrame:
    """
    Renames specified columns in a dataframe based on user input.

    Parameters
    ----------
    df_name : `str`
        Name of the dataframe.
    df : `DataFrame`
        Dataframe whose columns will be renamed.
    cols : `List[str]`
        List of column names to rename.

    Returns
    -------
    `DataFrame`
        Dataframe with renamed columns.
    """
    if not cols:
        return df

    print(f"Rename the columns of {df_name}")
    string_list = [input(f"Enter name for {name}: ") for name in cols]
    dict_rename = dict(zip(cols, string_list))
    df = df.rename(columns=dict_rename)
    print("Columns renamed!")
    return df


def merge_by_similarity(
    df_list: List[DataFrame],
    col_list: List[str],
    dist_min: int = 2,
    match_cols: int = 2,
    merge_mode: bool = False,
    manually: bool = False,
    drop_empty: bool = False,
    stdout: Any = sys.stdout,
) -> Tuple[List[DataFrame], List[str], ndarray]:
    """
    Merges dataframes based on column similarity using the Levenshtein distance.

    Parameters
    ----------
    df_list : `List`[`DataFrame`]
        The list of dataframes to be used in the process.
    col_list : `List[str]`
        The list of dataframe names corresponding to `df_list`.
    dist_min : `int`, default=2
        Minimum distance to determine that columns are similar.
    match_cols : `int`, default=2
        Minimum number of matching columns required for merging.
    merge_mode : `bool`, default=False
        If True, performs a left join on the largest dataframe with others having enough matching columns.
    manually : `bool`, default=False
        If False, avoids manual input when there are differences in columns.
    drop_empty : `bool`, default=False
        If True, discards frames with few columns and rows.

    Returns
    -------
    `Tuple`[`List`[`DataFrame`], `List[str]`, `ndarray`]
        - List of merged dataframes.
        - List of names corresponding to the merged dataframes.
        - Matrix of column similarity distances.
    """
    if drop_empty:
        df_list, col_list = check_empty_df(df_list, col_list)

    mtx = cal_cols_similarity(col_list)
    new_df_list = []
    new_col_list = []
    idx_to_exclude = []
    full_col_list_idx = list(range(len(col_list)))
    count = 0

    for idx in progressbar(range(len(col_list)), "Computing: ", out=stdout):
        for i in full_col_list_idx:
            if idx != i and i not in idx_to_exclude:
                if lev_dist(col_list[idx], col_list[i]) < dist_min:
                    cols_x = set(df_list[idx].columns)
                    cols_y = set(df_list[i].columns)
                    cols = list(cols_x.intersection(cols_y))

                    # Handle mismatched columns
                    if len(cols_x) != len(cols_y):
                        warning_type = "UserWarning"
                        msg = "You may have missed some of the columns.\n"
                        print(f"{warning_type}: {msg}")

                        if manually:
                            diff_x = [c for c in cols_x - cols_y]
                            diff_y = [c for c in cols_y - cols_x]
                            print("The columns that will be lost in each DataFrame are : \n")
                            print(f"Columns in {col_list[idx]}: {diff_x}")
                            print(f"\nColumns in {col_list[i]}: {diff_y}")

                            rename_manually = input("You want to rename the columns [y/n]: ")
                            table_data = create_table_tabular(df_list[idx], df_list[i])
                            headers = [col_list[idx], col_list[i]]
                            print(
                                "########################################################################################"
                            )
                            print("The total number of columns in each DataFrame is : ")
                            print(tabulate(table_data, headers=headers, tablefmt="grid"))

                            if rename_manually == "y":
                                df_list[idx] = rename_cols_dict(col_list[idx], df_list[idx], diff_x)
                                df_list[i] = rename_cols_dict(col_list[i], df_list[i], diff_y)
                                cols_x = set(df_list[idx].columns)
                                cols_y = set(df_list[i].columns)
                                cols = list(cols_x.intersection(cols_y))

                    # Merge dataframes based on matching columns
                    if len(cols) >= match_cols:
                        try:
                            df_list[idx] = pd.concat([df_list[idx][cols], df_list[i][cols]])
                            idx_to_exclude.append(i)
                        except pd.errors.InvalidIndexError as e:
                            warning_type = "ValueError"
                            msg = f"Invalid Index Error encountered while merging dataframes. {e}."
                            print(f"{warning_type}: {msg}")
                            sys.exit("Please check your column names.")
                    else:
                        warning_type = "UserWarning"
                        msg = f"DataFrame '{col_list[idx]}' does not contain enough matching columns.\n"
                        print(f"{warning_type}: {msg}")

                        if merge_mode and len(cols) > 0:
                            try:
                                clearConsole()
                                clear_output(wait=True)
                                print(count, "| Progress :", "{:.2%}".format(count / len(col_list)))

                                if len(df_list[idx]) > len(df_list[i]):
                                    print("Merging dataframes by left join")
                                    print(f"{col_list[idx]} | {col_list[i]}")
                                    print(
                                        f"Total Size df1: {len(df_list[idx]):,} "
                                        f"| Total Size df2: {len(df_list[i]):,}"
                                    )
                                    df_list[idx] = (
                                        df_list[idx]
                                        .merge(df_list[i].drop_duplicates(), how="left")
                                        .drop_duplicates()
                                    )
                                    df_list[idx] = rename_cols(df_list[idx])
                                    idx_to_exclude.append(i)
                                print("Merged")
                            except Exception as e:
                                print(f"An error occurred during merging: {e}")

        if idx not in idx_to_exclude:
            new_df_list.append(df_list[idx].reset_index(drop=True))
            new_col_list.append(col_list[idx])
        count += 1
    return new_df_list, new_col_list, mtx
