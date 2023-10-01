from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


class QualityAssesment:
    """Class containing the methods to generate the quality report of a set of tables."""

    def __init__(self, dfs: List[DataFrame]):
        self.dfs = dfs

    def get_report(
        self,
        df_names: List[str],
        report_name: str = "./report-health-checker.html",
        encoding: str = "latin1",
    ) -> DataFrame:
        """Function that returns the report generated

        Args:
            df_names (List[str]): `list` of dataframes, which are the tables.
            report_name (`str`, optional): name and path to be used to save the report. By default is set to `./report-health-checker.html`.
            encoding (`str`, optional): type of report encoding. By default is set to `latin1`.

        Returns:
            `DataFrame`: `DataFrame` of the generated report
        """
        df_sheet_files_info = self._iterative_evaluation(df_names)
        df_sheet_files_info.to_html(report_name, index=False, encoding=encoding)

        self.report = df_sheet_files_info
        return df_sheet_files_info

    def _iterative_evaluation(self, df_names: List[str]) -> DataFrame:
        """Function that iterates over the set of tables to build the report

        Args:
            df_names (List[`str`]): `list` of names of the tables on which it iterates.

        Returns:
            `DataFrame`: report generated from the set of tables.
        """
        df_sheet_files_info = pd.DataFrame()
        for i, df in enumerate(self.dfs):
            info = []
            for col in df.columns:
                nrows = df.isnull().sum()[col] + df[col].count()
                nrows_missing = df.isnull().sum()[col]
                percentage = nrows_missing / nrows
                datatype = df.dtypes[col]
                unique_vals = len(df[col].unique())
                info.append(
                    [
                        col,
                        datatype,
                        df_names[i],
                        nrows,
                        nrows_missing,
                        percentage,
                        unique_vals,
                    ]
                )
            info = np.array(info).reshape((-1, 7))
            info = pd.DataFrame(
                info,
                columns=[
                    "column name",
                    "data type",
                    "database name",
                    "# rows",
                    "# missing rows",
                    "# missing rows (percentage)",
                    "unique values",
                ],
            )
            info["# missing rows (percentage)"] = info["# missing rows (percentage)"].apply(
                lambda x: "{:.2%}".format(float(x))
            )
            info["# rows"] = info["# rows"].apply(lambda x: "{:,}".format(int(x)))
            info["# missing rows"] = info["# missing rows"].apply(lambda x: "{:,}".format(int(x)))
            info["unique values"] = info["unique values"].apply(lambda x: "{:,}".format(int(x)))
            df_sheet_files_info = pd.concat([df_sheet_files_info, info])

        return df_sheet_files_info


def check_empty_df(
    dfs: List[DataFrame], names: List[str], num_cols: int = 2
) -> Tuple[List[DataFrame], List[str]]:
    """Check if the `DataFrame` is empty or not

    Args:
        dfs (List[`DataFrame`]): List of dataframes to iterate over
        names (List[`str`]): List of `DataFrame` names
        num_cols (`int`): minimum number of columns of a `DataFrame`. By default is set to `2`

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


if __name__ == "__main__":
    # Create a DataFrame
    data = {"Name": ["John", "Alice", "Bob"], "Age": [25, 30, 35]}
    df = pd.DataFrame(data)
    table_name = "test_table"
    handler = QualityAssesment([df])
    handler.get_report([table_name])
