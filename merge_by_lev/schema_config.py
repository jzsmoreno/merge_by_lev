import json

import pandas as pd
import pyarrow
import pyarrow as pa
import yaml
from pyarrow import Table
from pydbsmgr.lightest import *
from pydbsmgr.main import *
from pydbsmgr.main import DataFrame
from pydbsmgr.utils.azure_sdk import *


class StandardColumns:
    """Allows columns to be transformed according to `SQL` standards
    or creates a `.json` file with the obfuscated columns."""

    def __init__(self, df: DataFrame) -> None:
        self.df = df.copy()

    def get_frame(
        self,
        json_name: str = "output.json",
        write_to_cloud: bool = False,
        connection_string: str = "",
        container_name: str = "",
        overwrite: bool = True,
        encoding: str = "utf-8",
        get_standard: bool = True,
        **kwargs,
    ) -> DataFrame:
        """Returns the `DataFrame` with the obfuscated columns or SQL standard format.

        Args
        ----
            json_name : `str`, optional
                Name of the dictionary .json file. By default it is set to output.json.
            write_to_cloud : `bool`, optional
                Boolean variable to write to an Azure storage account. By default it is set to False.
            connection_string : `str`, optional
                The connection string to storage account. By default it is set to "".
            container_name : `str`, optional
                Azure container name. By default it is set to "".
            overwrite : `bool`, optional
                Boolean variable that indicates whether to overwrite. By default it is set to True.
            encoding : `str`, optional
                File coding. By default it is set to `utf-8`.
            get_standard : `bool`, optional
                Instead of obfuscation returns the columns with SQL standards. By default it is set to True.

        Returns
        -------
            `DataFrame`: 
                DataFrame with changed columns

        Keyword Arguments
        ----------
            snake_case : `bool`, optional
                If true - transforms column names into snake case otherwise camel case will be used. Default is True.
            sort :`bool`, optional
                If true - sorts columns by their names in alphabetical order. Default is False.
            surrounding : `bool`, optional
                If true - removes brackets from column names before transformation. Default is True.
        """
        self._generate_dict(encoding)
        self._writer(json_name, write_to_cloud, connection_string, container_name, overwrite)
        if get_standard:
            df_renamed = self._sql_standards(**kwargs)
        else:
            df_renamed = (self.df).rename(columns=self.obfuscator)
        return df_renamed

    def _sql_standards(
        self, snake_case: bool = True, sort: bool = False, surrounding: bool = True
    ) -> DataFrame:
        """Transforms all column names into SQL standard format.

        Args
        ----
            snake_case : `bool`, optional
                If true - transforms column names into snake case otherwise camel case will be used. Default is `True`.
            sort : `bool`, optional
                If true - sorts columns by their names in alphabetical order. Default is `False`.
            surrounding : `bool`, optional
                If true - removes brackets from column names before transformation. Default is `True`.

        Returns
        -------
            `DataFrame`: 
                `DataFrame` with transformed columns.

        """
        df = (self.df).copy()
        df.columns = [
            col[1:-1] if col.startswith("[") and col.endswith("]") else col for col in df.columns
        ]
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace("_+", " ", regex=True)
        df.columns = df.columns.str.title()
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace(" ", "")
        df.columns = df.columns.str.replace("\n", "_")
        if snake_case:
            df.columns = [self._camel_to_snake(col) for col in df.columns]
        df.columns = df.columns.str[:128]
        if sort:
            df = self._sort_columns_by_length(df)
        if surrounding:
            df.columns = [f"[{col}]" for col in df.columns]
        return df

    def _sort_columns_by_length(self, dataframe: DataFrame) -> DataFrame:
        """Get the column names and sort them by length"""
        sorted_columns = sorted(dataframe.columns, key=len, reverse=True)
        sorted_dataframe = dataframe[sorted_columns]

        return sorted_dataframe

    def _camel_to_snake(self, column_name: str) -> str:
        """Use regular expression to convert camelCase/PascalCase to snake_case"""
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", column_name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def _writer(
        self,
        json_name: str,
        write_to_cloud: bool,
        connection_string: str,
        container_name: str,
        overwrite: bool,
    ) -> None:
        """Writer of the json file.

        Args
        ----
            json_name : `str`
                Name of the dictionary `.json` file.
            write_to_cloud : `bool`
                Boolean variable to write to an Azure storage account.
            connection_string : `str`
                The connection string to storage account.
            container_name : `str`
                Azure container name.
            overwrite : `bool`
                Boolean variable that indicates whether to overwrite.
        """
        if write_to_cloud:
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service_client.get_container_client(container_name)

            blob_client.upload_blob(
                name=json_name,
                data=self.json_bytes,
                overwrite=overwrite,
            )

    def _generate_dict(self, encoding: str) -> dict:
        """Generates the dictionary that renames the columns of the `DataFrame`.

        Args
        ----
            encoding : `str`
                File coding.

        Returns
        -------
            `dict`: 
                Dictionary to rename columns.
        """
        values = []
        keys = []
        for i, key in enumerate((self.df).columns):
            col_name = "[col_%s]" % i
            values.append(col_name)
            keys.append(key)
        obfuscator = dict(zip(keys, values))
        self.obfuscator = obfuscator
        self.inverted_dict = {v: k for k, v in (self.obfuscator).items()}
        self.json_string = json.dumps(self.inverted_dict)
        self.json_bytes = (self.json_string).encode(encoding)
        return obfuscator


class DataFrameToYaml:
    """Creates a `yaml` configuration file for database data type validation."""

    def __init__(self, df: DataFrame) -> None:
        self.df = df.copy()

    def create_yaml(
        self,
        dabase_name: str = "database",
        yaml_name: str = "output.yml",
        write_to_cloud: bool = False,
        connection_string: str = "",
        container_name: str = "",
        overwrite: bool = True,
    ) -> str:
        """Function that generates the schema of a `DataFrame` in a `.yml` file.

        Args:
        ----------
            dabase_name : `str`, optional
                Dataframe name. By default it is set to database.
            yaml_name : `str`, optional
                Output name of the .yml file. By default it is set to output.yml.
            write_to_cloud : `bool`, optional
                Boolean type variable indicating whether or not to write to the cloud. By default it is set to False.
            connection_string : `str`, optional
                Storage account and container connection string. By default it is set to "".
            container_name : `str`, optional
                Name of the container inside the storage account. By default it is set to "".
            overwrite : `bool`, optional 
                Boolean variable indicating whether the file is overwritten or not. By default it is set to True.
        """
        self.df.columns = [
            c.replace(" ", "_") for c in list(self.df.columns)
        ]  # Remove spaces from column names
        df_info = self._create_info()
        df_info["data type"] = [str(_type) for _type in df_info["data type"].to_list()]
        df_info["sql name"] = df_info["column name"]

        data = {}
        for col_name, data_type, sql_name in zip(
            df_info["column name"].to_list(),
            df_info["data type"],
            df_info["sql name"].to_list(),
        ):
            data[col_name] = {"type": [data_type], "sql_name": [sql_name]}

        yaml_data = yaml.dump({dabase_name: data})
        self.yaml_data = yaml_data
        yaml_bytes = yaml_data.encode()

        if write_to_cloud:
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service_client.get_container_client(container_name)

            blob_client.upload_blob(
                name=yaml_name,
                data=yaml_bytes,
                overwrite=overwrite,
            )
        else:
            with open(yaml_name, "w") as file:
                file.write(yaml_data)

        return self.yaml_data

    def _create_info(self) -> DataFrame:
        """Function that creates the column name and data type info from the `DataFrame`."""
        info = []

        for col in self.df.columns:
            datatype = self.df.dtypes[col]
            info.append([col, datatype])

        info = np.array(info).reshape((-1, 2))
        info = pd.DataFrame(
            info,
            columns=["column name", "data type"],
        )

        return info


def recursive_correction(df_: DataFrame, input_string: str, unchanged_names: List[str]) -> Table:
    pattern = r"Conversion failed for column (\w+) with type"
    match = re.search(pattern, input_string)
    column_name = match.group(1)
    msg = "the column {%s} will be converted to text" % column_name
    print(msg)
    df_[column_name] = df_[column_name].astype(str)
    try:
        data_handler = DataSchema(df_)
        return (pa.Table.from_pandas(df_, schema=data_handler.get_schema())).rename_columns(
            unchanged_names
        )
    except (pa.lib.ArrowTypeError, pyarrow.lib.ArrowInvalid) as e:
        iteration_match = re.search(pattern, (str(e).split(","))[-1])
        iteration_column_name = iteration_match.group(1)
        if column_name != iteration_column_name:
            return recursive_correction(df_, (str(e).split(","))[-1], unchanged_names)


class DataSchema(DataFrameToYaml):
    def __init__(self, df: DataFrame):
        super().__init__(df)
        self.cols = df.columns.to_list()

    def get_schema(
        self,
        format_type="pyarrow",
        dabase_name: str = "database",
        yaml_name: str = "output.yml",
        write_to_cloud: bool = False,
        connection_string: str = "",
        container_name: str = "",
        overwrite: bool = True,
    ):
        if format_type == "yaml":
            return self.create_yaml(
                dabase_name, yaml_name, write_to_cloud, connection_string, container_name, overwrite
            )
        elif format_type == "pyarrow":
            self.create_yaml(
                dabase_name, yaml_name, write_to_cloud, connection_string, container_name, overwrite
            )
            with open(yaml_name, "r") as file:
                data = yaml.safe_load(file)
            fields = []

            for col_name, col_info in data["database"].items():
                col_type = col_info["type"][0]
                if col_type == "int64":
                    dtype = pa.int64()
                elif col_type == "object":
                    dtype = pa.string()
                elif col_type == "float64":
                    dtype = pa.float64()
                elif col_type == "datetime64[ns]":
                    dtype = pa.timestamp("ns")
                elif col_type == "int32":
                    dtype = pa.int32()
                elif col_type == "bool":
                    dtype = pa.bool_()
                else:
                    raise ValueError(f"Unsupported type: {col_type}")

                fields.append((col_info["sql_name"][0], dtype))

        schema = pa.schema(fields)

        schema_dict = {
            "schema": {
                "fields": [
                    {"name": field.name, "type": field.type.to_pandas_dtype()} for field in schema
                ]
            }
        }

        with open("schema.yml", "w") as file:
            yaml.dump(schema_dict, file)

        output_file_path = yaml_name
        schema_file_path = "schema.yml"

        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        if os.path.exists(schema_file_path):
            os.remove(schema_file_path)

        try:
            schema = pa.Schema.from_pandas(self.df)
        except:
            None
        self.schema = schema

        return schema

    def get_table(self) -> Table:
        try:
            table = pa.Table.from_pandas(self.df, schema=self.get_schema())
        except:
            try:
                record_batch = pa.RecordBatch.from_pandas(self.df)
                table = pa.Table.from_pandas(self.df)
            except (pa.lib.ArrowTypeError, pyarrow.lib.ArrowInvalid) as e:
                warning_type = "UserWarning"
                msg = "It was not possible to create the table\n"
                msg += "Error: {%s}" % e
                print(f"{warning_type}: {msg}")
                return recursive_correction(self.df, (str(e).split(","))[-1], self.cols)

        return table.rename_columns(self.cols)


if __name__ == "__main__":
    # Create a DataFrame
    data = {
        "Name": ["Dani", "John", "Alice", "Bob"],
        "Age": ["32", 25, 30, 35],
        "Points value": ["0", 1, 2, 3],
    }
    df = pd.DataFrame(data)
    table_name = "test_table"
    data_handler = DataSchema(df)
    schema = data_handler.get_schema()
    table = data_handler.get_table()
    column_handler = StandardColumns(df)
    df = column_handler.get_frame(surrounding=False, snake_case=False)
    breakpoint()
