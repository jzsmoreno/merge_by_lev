import os
import re
import sys
from functools import lru_cache
from io import TextIOWrapper
from typing import List, Tuple

import numpy as np
import pandas as pd
from IPython.display import clear_output
from numpy import ndarray
from pandas.core.frame import DataFrame
from tabulate import tabulate


def progressbar(
    it: range, prefix: str = "", size: int = 40, out: TextIOWrapper = sys.stdout
) -> None:
    """
    Auxiliary function displaying a progress bar
    """
    count = len(it)

    def show(j):
        x = int(size * j / count)
        print(
            "{}[{}{}] {}/{}".format(prefix, "#" * x, "." * (size - x), j, count),
            end="\r",
            file=out,
            flush=True,
        )

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)


def clearConsole() -> None:
    """
    Auxiliary function that cleans the console
    """
    command = "clear"
    if os.name in ("nt", "dos"):
        command = "cls"
    os.system(command)


def check_cols_to_match(dict_dfs: dict[DataFrame], df_names: List[DataFrame]) -> None:
    """
    Receives a dictionary of dataframes (dict_dfs) and a list of dataframe names (dfs_names).
    Then check if the dataframes have the same columns. Print the data frames that do not match

    params:
        dict_dfs (Dictionary) : Contains the dataframes to be analyzed
        df_names (`List`) : Contains the keys (names of each dataframe) of the dictionary

    returns:
        This function returns a summary of the condition of the columns

    example:
        # dfs -> (`List` of dataframes)
        # names -> (`List` of names)
        dict_dfs = {name:df for df, name in zip(dfs, names)}
        check_cols_to_match(dict_dfs, df_names)
        >>
    """
    cols_set = set([col for name in df_names for col in dict_dfs[name].columns])
    for name in df_names:
        col_set = set(dict_dfs[name].columns)
        if col_set == cols_set:
            print(name + " -> Passed")
        else:
            print(name + " -> Not Passed")
            print(f"Columns do not match -> {cols_set - col_set}")
            print(f"Original columns -> {col_set}")
            print("\n")


# https://github.com/jzsmoreno/Workflow.git
# auxiliary function to rename columns after each match
def rename_cols(df: DataFrame) -> DataFrame:
    """
    Operates on a dataframe resulting from a join.
    Identifying the cases in which there was a renaming of similar columns
    with different information, consolidating them.

    params:
        df (`Dataframe`) : The dataframe on which you want to operate

    returns:
        df (`Dataframe`) : The same df dataframe with the consolidated columns

    example:
        df_1 = df_1.merge(df_2, how = 'left')
        df_1 = rename_cols(df_1)
        >>
    """
    cols = []
    for i in df.columns:
        cols.append(i.replace("_x", ""))
        cols.append(i.replace("_y", ""))

    cols = [*set(cols)]

    for i in cols:
        try:
            df[i + "_x"] = df[i + "_x"].fillna(df[i + "_y"])
            df = df.drop(columns=[i + "_y"])
            df.rename(columns={i + "_x": i}, inplace=True)
        except:
            None

    return df


def clean_names(x: str, pattern: str = r"[a-zA-Zñáéíóú_]+\b") -> str:
    """
    Receives a string for cleaning to be used in merge_by_similarity function.

    params:
        x (String) : Character string to which a regular expression is to be applied
        pattern (regex) : By default extracts names without numerical characters

    returns:
        result (String) : The clean text string

    example:
        x = 'stamp_1'
        clean_names(x)
        >> 'stamp'
    """
    result = re.findall(pattern, str(x).replace("_", ""))
    if len(result) > 0:
        result = "_".join(result)
    else:
        pattern = r"[a-zA-Z]+"
        result = re.findall(pattern, str(x).replace("_", ""))
        result = "_".join(result)
    return result


def lev_dist(a: str, b: str) -> int:
    """
    This function will calculate the levenshtein distance between two input
    strings a and b

    params:
        a (String) : The first string you want to compare
        b (String) : The second string you want to compare

    returns:
        This function will return the distnace between string a and b.

    example:
        a = 'stamp'
        b = 'stomp'
        lev_dist(a,b)
        >> 1.0
    """

    @lru_cache(None)  # for memorization
    def min_dist(s1, s2):
        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        # no change required
        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),  # insert character
            min_dist(s1 + 1, s2),  # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )

    return min_dist(0, 0)


def cal_cols_similarity(col_list: List[str]) -> ndarray:
    """
    Calculate in pairs the levenshtein distance of the chars according to their name

    params:
        col_list (`List`) : List with the chars names

    returns:
        mtx (`np.array`) : Matrix of $n$ x $n$ containing the results for $n$ chars.

    example:
        cal_cols_similarity(col_list)
        >>
    """
    n = len(col_list)
    mtx = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            mtx[i, j] = lev_dist(col_list[i], col_list[j])
    return mtx


def create_table_tabular(df1: DataFrame, df2: DataFrame) -> List[List[str]]:
    """Create a table for column names from two dataframes

    Args:
        df1 (`DataFrame`): First dataframe
        df2 (`DataFrame`): Second dataframe

    Returns:
        List[List[`str`]]: List of rows for each of the columns of both dataframes
    """
    table = []
    col_names_df1 = df1.columns
    col_names_df2 = df2.columns
    max_length = max(len(col_names_df1), len(col_names_df2))
    for i in range(max_length):
        col1 = col_names_df1[i] if i < len(col_names_df1) else ""
        col2 = col_names_df2[i] if i < len(col_names_df2) else ""
        table.append([col1, col2])
    return table


def rename_cols_dict(df_name: str, df: DataFrame, cols: list) -> DataFrame:
    """Function that allows to rename a segment of columns of a dataframe from a list as input.

    Args:
        df_name (str): Name of dataframe
        df (DataFrame): Dataframe whose columns names will be changed
        cols (list): List indicating the names of the columns to be changed

    Returns:
        DataFrame: Processed dataframe with changed names
    """
    if not cols:
        return df
    else:
        print(f"Rename the columns of {df_name}")
        string_list = []
        for name in cols:
            string = input(f"Enter name for {name}: ")
            string_list.append(string)
        # Create a dictionary with keys in cols and values in string_list
        dict_rename = dict(zip(cols, string_list))
        # Rename all the columns using the new dictionary created above
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
) -> Tuple[List[DataFrame], List[str], ndarray]:
    """
    It makes use of the levenshtein distance to calculate
    a similarity between dataframes according to a list of names
    to concatenate them or make a left join (if merge_mode = `True`).

    params:
        df_list (List of Dataframes) : The list of dataframes to be used in the process
        col_list (List of chars) : The list of dataframe names
        dist_min (`int`) : Minimum distance to determine that they are equal. By default is set to `2`.
        match_cols (`int`) : Minimum number of columns to concatenate. By default is set to `2`.
        merge_mode (Boolean) : If `True`, it seeks to take the largest dataframe and make a left join with those that share columns with each other.
        manually (Boolean) : If `False` avoids inputs when there are differences in columns. By default is set to `False`.
    """
    mtx = cal_cols_similarity(col_list)
    new_df_list = []
    new_col_list = []
    idx_to_exclude = []
    full_col_list_idx = list(range(len(col_list)))
    count = 0
    for idx in progressbar(range(len(col_list)), "Computing: "):
        for i in full_col_list_idx:
            if idx != i:
                if i not in idx_to_exclude:
                    if lev_dist(col_list[idx], col_list[i]) < dist_min:
                        cols_x = set(df_list[idx].columns)
                        cols_y = set(df_list[i].columns)
                        cols = list(cols_x.intersection(cols_y))
                        if len(cols_x) != len(cols_y):
                            warning_type = "UserWarning"
                            msg = "You may have missed some of the columns.\n"
                            print(f"{warning_type}: {msg}")
                            if manually:
                                print("The columns that will be lost in each DataFrame are : \n")
                                diff_x = [c for c in cols_x - cols_y]
                                diff_y = [c for c in cols_y - cols_x]
                                print("Columns in", col_list[idx], ":", diff_x)
                                print("\n")
                                print("Columns in", col_list[i], ":", diff_y)
                                rename_manually = input("You want to rename the columns [y/n] : ")
                                table_data = create_table_tabular(df_list[idx], df_list[i])
                                headers = [col_list[idx], col_list[i]]
                                print(
                                    "########################################################################################"
                                )
                                print("The total number of columns in each DataFrame is : ")
                                print(tabulate(table_data, headers=headers, tablefmt="grid"))
                                if rename_manually == "y":
                                    df_list[idx] = rename_cols_dict(col_list[idx], df_list[idx], diff_x)
                                    print(
                                        "########################################################################################"
                                    )
                                    df_list[i] = rename_cols_dict(col_list[i], df_list[i], diff_y)
                                    cols_x = set(df_list[idx].columns)
                                    cols_y = set(df_list[i].columns)
                                    cols = list(cols_x.intersection(cols_y))

                        if len(cols) >= match_cols:
                            try:
                                df_list[idx] = pd.concat([df_list[idx][cols], df_list[i][cols]])
                                idx_to_exclude.append(i)
                            except:
                                None
                        else:
                            if merge_mode:
                                if len(cols) > 0:
                                    try:
                                        clearConsole()
                                        clear_output(wait=True)
                                        print(
                                            count,
                                            "| Progress :",
                                            "{:.2%}".format(count / len(col_list)),
                                        )
                                        if len(df_list[idx]) > len(df_list[i]):
                                            print("merging dataframes by left")
                                            print(col_list[idx], "|", col_list[i])
                                            print(
                                                "Total Size df1: ",
                                                "{:,}".format(len(df_list[idx])),
                                                "| Total Size df2: ",
                                                "{:,}".format(len(df_list[i])),
                                            )
                                            df_list[idx] = (
                                                df_list[idx].merge(
                                                    df_list[i].drop_duplicates(), how="left"
                                                )
                                            ).drop_duplicates()
                                            df_list[idx] = rename_cols(df_list[idx])
                                            idx_to_exclude.append(i)
                                        print("merged")
                                    except:
                                        None
        if idx not in idx_to_exclude:
            new_df_list.append(df_list[idx].reset_index(drop=True))
            new_col_list.append(col_list[idx])
        count += 1
    return new_df_list, new_col_list, mtx
