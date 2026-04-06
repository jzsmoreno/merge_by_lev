from typing import List, Tuple, Union

import pandas as pd
import polars as pl


def _to_polars(df: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    """
    Converts a DataFrame to Polars format.

    Parameters
    ----------
    df : `Union[pd.DataFrame, pl.DataFrame]`
        The DataFrame to convert.

    Returns
    -------
    `pl.DataFrame`
        The DataFrame in Polars format.
    """
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    return df


def _is_polars(df: Union[pd.DataFrame, pl.DataFrame]) -> bool:
    """Check if a DataFrame is a Polars DataFrame."""
    return isinstance(df, pl.DataFrame)


class QualityAssessment:
    """Class containing the methods to generate the quality report of a set of tables."""

    def __init__(self, dfs: List[Union[pd.DataFrame, pl.DataFrame]]):
        self.original_types = [_is_polars(df) for df in dfs]
        self.dfs = [_to_polars(df) for df in dfs]

    def get_report(
        self,
        df_names: List[str],
        report_name: str = "./report-health-checker.html",
        encoding: str = "latin1",
        return_polars: bool = False,
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Function that returns the report generated.

        Parameters
        ----------
        df_names : `List[str]`
            List of names for the dataframes.
        report_name : `str`, optional
            Name and path to be used to save the report. Defaults to "./report-health-checker.html".
        encoding : `str`, optional
            Encoding type of the report. Defaults to "latin1".
        return_polars : `bool`, optional
            If True, returns a Polars DataFrame. Defaults to False (returns Pandas).

        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame] :
            DataFrame containing the generated report.
        """
        df_sheet_files_info = self._iterative_evaluation(df_names)

        if not return_polars:
            df_sheet_files_info.to_pandas().to_html(report_name, index=False, encoding=encoding)
        else:
            df_sheet_files_info.write_html(report_name)

        self.report = df_sheet_files_info if return_polars else df_sheet_files_info.to_pandas()
        return self.report

    def _iterative_evaluation(self, df_names: List[str]) -> pl.DataFrame:
        """Function that iterates over the set of tables to build the report.

        Parameters
        ----------
        df_names : `List[str]`
            List of names of the tables on which it iterates.

        Returns
        -------
        pl.DataFrame :
            Report generated from the set of tables.
        """
        rows = []
        for i, df in enumerate(self.dfs):
            nrows_total = len(df)
            null_counts = df.null_count()
            for col in df.columns:
                nrows_missing = null_counts[col].item()
                percentage_missing = nrows_missing / nrows_total
                datatype = df.schema[col]
                unique_vals = df[col].n_unique()

                rows.append(
                    {
                        "column name": col,
                        "data type": str(datatype),
                        "database name": df_names[i],
                        "# rows": nrows_total,
                        "# missing rows": nrows_missing,
                        "# missing rows (percentage)": f"{percentage_missing:.2%}",
                        "unique values": f"{unique_vals:,}",
                    }
                )

        info_df = pl.DataFrame(
            rows,
            schema={
                "column name": pl.String,
                "data type": pl.String,
                "database name": pl.String,
                "# rows": pl.Int64,
                "# missing rows": pl.Int64,
                "# missing rows (percentage)": pl.String,
                "unique values": pl.String,
            },
        )

        return info_df


def check_empty_df(
    dfs: List[Union[pd.DataFrame, pl.DataFrame]], names: List[str], num_cols: int = 2
) -> Tuple[List[Union[pd.DataFrame, pl.DataFrame]], List[str]]:
    """Check if the `DataFrame` is empty or not.

    Parameters
    ----------
    dfs : `List[Union[pd.DataFrame, pl.DataFrame]]`
        List of dataframes to iterate over.
    names : `List[str]`
        List of DataFrame names.
    num_cols : `int`, optional
        Minimum number of columns of a DataFrame. Defaults to 2.

    Returns
    -------
    Tuple[List[Union[pd.DataFrame, pl.DataFrame]], List[str]] :
        Verified dataframes and names.
    """
    verified_dfs = []
    verified_names = []

    for df, name in zip(dfs, names):
        if len(df.columns) > num_cols and len(df) > num_cols:
            verified_dfs.append(df)
            verified_names.append(name)

    return verified_dfs, verified_names


if __name__ == "__main__":
    data = {"Name": ["John", "Alice", "Bob"], "Age": [25, 30, 35]}
    df_pandas = pd.DataFrame(data)
    df_polars = pl.DataFrame(data)
    table_name = "test_table"
    handler = QualityAssessment([df_pandas, df_polars])
    handler.get_report([table_name + "_1", table_name + "_2"])
