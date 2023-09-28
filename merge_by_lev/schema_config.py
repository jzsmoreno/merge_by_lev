import json

import pyarrow as pa
import yaml
from pydbsmgr.main import *
from pydbsmgr.main import DataFrame
from pydbsmgr.utils.azure_sdk import *
from pydbsmgr.utils.azure_sdk import DataFrame


class ColumnObfuscation:
    """Allows generic column renaming and creates a `.json` file with the equivalent names."""

    def __init__(self, df: DataFrame) -> None:
        self.df = df.copy()

    def get_frame(
        self,
        json_name: str = "output.json",
        write_to_cloud: bool = True,
        connection_string: str = "",
        container_name: str = "",
        overwrite: bool = True,
        encoding: str = "utf-8",
    ) -> DataFrame:
        """Returns the `DataFrame` with the obfuscated columns.

        Args:
            json_name (`str`, optional): name of the dictionary `.json` file. By default it is set to `output.json`.
            write_to_cloud (`bool`, optional): boolean variable to write to an Azure storage account. By default it is set to `True`.
            connection_string (`str`, optional): the connection string to storage account. By default it is set to "".
            container_name (`str`, optional): Azure container name. By default it is set to "".
            overwrite (`bool`, optional): boolean variable that indicates whether to overwrite. By default it is set to `True`.
            encoding (`str`, optional): file coding. By default it is set to `utf-8`.

        Returns:
            `DataFrame`: `DataFrame` with changed columns
        """
        self._generate_dict(encoding)
        self._writer(json_name, write_to_cloud, connection_string, container_name, overwrite)
        df_renamed = (self.df).rename(columns=self.obfuscator)
        return df_renamed

    def _writer(
        self,
        json_name: str,
        write_to_cloud: bool,
        connection_string: str,
        container_name: str,
        overwrite: bool,
    ) -> None:
        """writer of the json file.

        Args:
            json_name (`str`): name of the dictionary `.json` file.
            write_to_cloud (`bool`): boolean variable to write to an Azure storage account.
            connection_string (`str`): the connection string to storage account.
            container_name (`str`): Azure container name.
            overwrite (`bool`): boolean variable that indicates whether to overwrite.
        """
        if write_to_cloud:
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service_client.get_container_client(container_name)

            blob_client.upload_blob(
                name=json_name,
                data=self.json_bytes,
                overwrite=overwrite,
            )
        else:
            with open(json_name, "w") as file:
                file.write(self.json_string)

    def _generate_dict(self, encoding: str) -> dict:
        """generates the dictionary that renames the columns of the `DataFrame`.

        Args:
            encoding (`str`): file coding.

        Returns:
            `dict`: dictionary to rename columns.
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
            dabase_name (`str`, optional): `Dataframe` name. By default it is set to `database`
            yaml_name (`str`, optional): output name of the `.yml` file. By default it is set to `output.yml`
            write_to_cloud (`bool`, optional): boolean type variable indicating whether or not to write to the cloud. By default it is set to `False`
            connection_string (`str`, optional): storage account and container connection string. By default it is set to `""`.
            container_name (`str`, optional): name of the container inside the storage account. By default it is set to `""`.
            overwrite (`bool`, optional): boolean variable indicating whether the file is overwritten or not. By default it is set to `True`.
        """
        df_info = self._create_info()
        df_info["data type"] = [str(_type) for _type in df_info["data type"].to_list()]
        df_info["sql name"] = [col_name.replace(" ", "_") for col_name in df_info["column name"]]

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


class DataSchema(DataFrameToYaml):
    def __init__(self, df: DataFrame):
        super().__init__(df)

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

        self.schema = schema
        return schema

    def get_table(self):
        try:
            table = pa.Table.from_pandas(self.df, schema=self.get_schema())
        except:
            try:
                record_batch = pa.RecordBatch.from_pandas(self.df)
                table = pa.Table.from_pandas(self.df)
            except:
                try:
                    self.df = (self.df).convert_dtypes()
                    table = pa.Table.from_pandas(self.df)
                except pa.lib.ArrowTypeError as e:
                    warning_type = "UserWarning"
                    msg = "It was not possible to create the table\n"
                    msg += "Error: {%s}" % e
                    return print(f"{warning_type}: {msg}")
        return table


if __name__ == "__main__":
    # Create a DataFrame
    data = {"Name": ["John", "Alice", "Bob"], "Age": [25, 30, 35]}
    df = pd.DataFrame(data)
    table_name = "test_table"
    data_handler = DataSchema(df)
    # data_handler.get_schema()
    table = data_handler.get_table()
    column_handler = ColumnObfuscation(df)
    df = column_handler.get_frame(write_to_cloud=False)
    breakpoint()
