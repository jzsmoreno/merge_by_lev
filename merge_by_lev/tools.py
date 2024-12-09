from typing import List, Tuple

import numpy as np
import pandas as pd


class QualityAssessment:
    """Class containing the methods to generate the quality report of a set of tables."""

    def __init__(self, dfs: List[pd.DataFrame]):
        self.dfs = dfs

    def get_report(
        self,
        df_names: List[str],
        report_name: str = "./report-health-checker.html",
        encoding: str = "latin1",
    ) -> pd.DataFrame:
        """Function that returns the report generated.

        Parameters
        ----------
        df_names : `List[str]`
            List of names for the dataframes.
        report_name : `str`, optional
            Name and path to be used to save the report. Defaults to "./report-health-checker.html".
        encoding : `str`, optional
            Encoding type of the report. Defaults to "latin1".

        Returns
        -------
        pd.DataFrame :
            DataFrame containing the generated report.
        """
        df_sheet_files_info = self._iterative_evaluation(df_names)
        df_sheet_files_info.to_html(report_name, index=False, encoding=encoding)

        self.report = df_sheet_files_info
        return df_sheet_files_info

    def _iterative_evaluation(self, df_names: List[str]) -> pd.DataFrame:
        """Function that iterates over the set of tables to build the report.

        Parameters
        ----------
        df_names : `List[str]`
            List of names of the tables on which it iterates.

        Returns
        -------
        pd.DataFrame :
            Report generated from the set of tables.
        """
        rows = []
        for i, df in enumerate(self.dfs):
            nrows_total = len(df)
            null_counts = df.isnull().sum()
            unique_counts = df.nunique()

            for col in df.columns:
                nrows_missing = null_counts[col]
                percentage_missing = nrows_missing / nrows_total
                datatype = df.dtypes[col]
                unique_vals = unique_counts[col]

                rows.append(
                    [
                        col,
                        datatype,
                        df_names[i],
                        nrows_total,
                        nrows_missing,
                        percentage_missing,
                        unique_vals,
                    ]
                )

        info_df = pd.DataFrame(
            rows,
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

        # Apply formatting
        info_df["# missing rows (percentage)"] = info_df["# missing rows (percentage)"].apply(
            lambda x: "{:.2%}".format(x)
        )
        info_df["# rows"] = info_df["# rows"].apply(lambda x: f"{x:,}")
        info_df["# missing rows"] = info_df["# missing rows"].apply(lambda x: f"{x:,}")
        info_df["unique values"] = info_df["unique values"].apply(lambda x: f"{x:,}")

        return info_df


def check_empty_df(
    dfs: List[pd.DataFrame], names: List[str], num_cols: int = 2
) -> Tuple[List[pd.DataFrame], List[str]]:
    """Check if the `DataFrame` is empty or not.

    Parameters
    ----------
    dfs : `List[pd.DataFrame]`
        List of dataframes to iterate over.
    names : `List[str]`
        List of DataFrame names.
    num_cols : `int`, optional
        Minimum number of columns of a DataFrame. Defaults to 2.

    Returns
    -------
    Tuple[List[pd.DataFrame], List[str]] :
        Verified dataframes and names.
    """
    verified_dfs = []
    verified_names = []

    # Iterar sobre los DataFrames y nombres
    for df, name in zip(dfs, names):
        if len(df.columns) > num_cols and len(df) > num_cols:
            verified_dfs.append(df)
            verified_names.append(name)

    return verified_dfs, verified_names


if __name__ == "__main__":
    # Create a DataFrame
    data = {"Name": ["John", "Alice", "Bob"], "Age": [25, 30, 35]}
    df = pd.DataFrame(data)
    table_name = "test_table"
    handler = QualityAssessment([df])
    handler.get_report([table_name])
