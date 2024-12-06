import json
import os
import re
from typing import List, Optional, Union

import pandas as pd
import pyarrow as pa
import yaml
from azure.storage.blob import BlobServiceClient


class StandardColumns:
    """Allows columns to be transformed according to SQL standards or creates a .json file with the obfuscated columns."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def get_frame(
        self,
        json_name: str = "output.json",
        write_to_cloud: bool = False,
        connection_string: Optional[str] = "",
        container_name: Optional[str] = "",
        overwrite: bool = True,
        encoding: str = "utf-8",
        get_standard: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Returns the DataFrame with obfuscated columns or SQL standard format."""
        self._generate_dict(encoding)
        if write_to_cloud:
            self._write_to_cloud(json_name, connection_string, container_name, overwrite)
        if get_standard:
            df_renamed = self._sql_standards(**kwargs)
        else:
            df_renamed = self.df.rename(columns=self.obfuscator)
        return df_renamed

    def _sql_standards(
        self,
        snake_case: bool = True,
        sort: bool = False,
        surrounding: bool = True,
    ) -> pd.DataFrame:
        """Transforms all column names into SQL standard format."""
        df = self.df.copy()
        if surrounding:
            df.columns = [col.strip("[]") for col in df.columns]
        df.columns = (
            df.columns.str.lower()
            .str.replace("_+", " ", regex=True)
            .str.title()
            .str.replace(" ", "")
            .str.replace("\n", "_")
            .map(lambda x: self._camel_to_snake(x) if snake_case else x)
            .str[:128]
        )
        return df.sort_index(axis=1) if sort else df

    def _camel_to_snake(self, column_name: str) -> str:
        """Convert camelCase/PascalCase to snake_case."""
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", column_name)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def _write_to_cloud(
        self,
        json_name: str,
        connection_string: Optional[str],
        container_name: Optional[str],
        overwrite: bool,
    ) -> None:
        """Upload JSON file to Azure Blob Storage."""
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_container_client(container_name)
        blob_client.upload_blob(name=json_name, data=self.json_bytes, overwrite=overwrite)

    def _generate_dict(self, encoding: str) -> None:
        """Generates a dictionary to rename DataFrame columns."""
        obfuscator = {col: f"[col_{i}]" for i, col in enumerate(self.df.columns)}
        self.obfuscator = obfuscator
        self.inverted_dict = {v: k for k, v in obfuscator.items()}
        self.json_string = json.dumps(self.inverted_dict)
        self.json_bytes = self.json_string.encode(encoding)


class DataFrameToYaml:
    """Creates a YAML configuration file for database data type validation."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def create_yaml(
        self,
        database_name: str = "database",
        yaml_name: str = "output.yml",
        write_to_cloud: bool = False,
        connection_string: Optional[str] = "",
        container_name: Optional[str] = "",
        overwrite: bool = True,
    ) -> str:
        """Function that generates the schema of a DataFrame in a .yml file."""
        self.df.columns = [c.replace(" ", "_") for c in self.df.columns]
        df_info = self._create_info()
        data = {
            col_name: {"type": _type, "sql_name": sql_name}
            for col_name, _type, sql_name in zip(
                df_info["column name"], df_info["data type"].apply(str), df_info["column name"]
            )
        }
        yaml_data = yaml.dump({database_name: data})
        yaml_bytes = yaml_data.encode()

        if write_to_cloud:
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service_client.get_container_client(container_name)
            blob_client.upload_blob(name=yaml_name, data=yaml_bytes, overwrite=overwrite)
        else:
            with open(yaml_name, "w") as file:
                file.write(yaml_data)

        return yaml_data

    def _create_info(self) -> pd.DataFrame:
        """Function that creates the column name and data type info from the DataFrame."""
        return pd.DataFrame(
            {"column name": self.df.columns, "data type": self.df.dtypes}, index=self.df.columns
        )


class DataSchema(DataFrameToYaml):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)

    def get_schema(
        self,
        format_type: str = "pyarrow",
        database_name: str = "database",
        yaml_name: str = "output.yml",
        write_to_cloud: bool = False,
        connection_string: Optional[str] = "",
        container_name: Optional[str] = "",
        overwrite: bool = True,
    ) -> Union[pd.DataFrame, pa.Schema]:
        if format_type == "yaml":
            return self.create_yaml(
                database_name,
                yaml_name,
                write_to_cloud,
                connection_string,
                container_name,
                overwrite,
            )
        elif format_type == "pyarrow":
            self.create_yaml(
                database_name,
                yaml_name,
                write_to_cloud,
                connection_string,
                container_name,
                overwrite,
            )
            with open(yaml_name, "r") as file:
                data = yaml.safe_load(file)

            fields = [
                (col_info["sql_name"], col_info["type"])
                for col_name, col_info in data[database_name].items()
            ]
            dtype_mapping = {
                "int8": pa.int8(),
                "int16": pa.int16(),
                "int64": pa.int64(),
                "int32": pa.int32(),
                "float16": pa.float16(),
                "float32": pa.float32(),
                "float64": pa.float64(),
                "bool": pa.bool_(),
                "object": pa.string(),
                "datetime64[ns]": pa.timestamp("ns"),
                "category": pa.string(),
            }
            fields = [
                (
                    col_info["sql_name"],
                    dtype_mapping[col_info["type"]],
                )
                for col_name, col_info in data[database_name].items()
            ]

            schema = pa.schema(fields)
            schema_dict = {
                "schema": {
                    "fields": [
                        {"name": field.name, "type": field.type.to_pandas_dtype()}
                        for field in schema
                    ]
                }
            }

            with open("schema.yml", "w") as file:
                yaml.dump(schema_dict, file)

            if os.path.exists(yaml_name):
                os.remove(yaml_name)

            if os.path.exists("schema.yml"):
                os.remove("schema.yml")

            return schema

        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def get_table(self) -> pa.Table:
        try:
            table = pa.Table.from_pandas(self.df, schema=self.get_schema())
        except (pa.lib.ArrowTypeError, pa.lib.ArrowInvalid) as e:
            return recursive_correction(self.df, str(e), list(self.df.columns))

        return table


def recursive_correction(
    df: pd.DataFrame, error_message: str, unchanged_names: List[str]
) -> pa.Table:
    pattern = r"Conversion failed for column (\w+) with type"
    match = re.search(pattern, error_message)
    if not match:
        raise ValueError(f"Could not find the column name in the error message: {error_message}")

    column_name = match.group(1)
    print(f"The column {column_name} will be converted to text.")
    df[column_name] = df[column_name].astype(str)
    try:
        data_handler = DataSchema(df)
        return pa.Table.from_pandas(df, schema=data_handler.get_schema()).rename_columns(
            unchanged_names
        )
    except (pa.lib.ArrowTypeError, pa.lib.ArrowInvalid) as e:
        new_match = re.search(pattern, str(e).split(",")[-1])
        if not new_match or column_name == new_match.group(1):
            raise e
        return recursive_correction(df, str(e).split(",")[-1], unchanged_names)


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
