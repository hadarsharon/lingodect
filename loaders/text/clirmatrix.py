import glob
from functools import cached_property
from pathlib import Path
from typing import Optional, Literal

import pandas as pd

from utils.config import Paths
from utils.datasets import BaseTextDataset


class CLIRMatrix(BaseTextDataset):
    DATA_DIR = Paths.DATASETS / "text/CLIRMatrix/data"
    DATA_EXT = ".parquet.gz"
    DATA_FILES = glob.glob(rf"{DATA_DIR}/*{DATA_EXT}")
    LANGUAGES: list[str] = sorted(set(Path(Path(file).stem).stem for file in DATA_FILES))
    SQLITE3_COLUMNS = ("language", "partition", "text")
    SQLITE3_TABLE = "text_clirmatrix"
    LANGUAGE_KEY = "language"
    TEXT_KEY = "text"
    PARTITION_KEY = "partition"
    PARTITIONS = ("train", "test")

    @staticmethod
    def read_dataframe(language: Optional[str] = None, columns: Optional[tuple[str, ...]] = None) -> pd.DataFrame:
        if language:
            try:  # fetch language-specific file
                [language_file] = filter(lambda f: Path(Path(f).stem).stem == language, CLIRMatrix.DATA_FILES)
            except ValueError:
                raise FileNotFoundError(
                    f"{CLIRMatrix.DATA_EXT} data file for language '{language}' not found in {CLIRMatrix.DATA_DIR}"
                ) from None  # suppress original exception as it's not useful
            else:
                df = pd.read_parquet(language_file, columns=list(columns) if columns else None)
        else:  # read all language(s) files
            df = pd.concat(
                (pd.read_parquet(file, columns=list(columns) if columns else None) for file in CLIRMatrix.DATA_FILES)
            )
        return df

    def get_base_dataset(self, partition: Optional[Literal["train", "test", "dev"]] = None) -> pd.DataFrame:
        if self.sqlite_conn:  # prefer sqlite as it's faster than reading on-the-fly
            sql = f"""
                    SELECT 
                        LOWER({self.language_key}) AS {self.language_key}, 
                        {self.text_key} AS {self.text_key} 
                    FROM {self.db_table}
                    {f"WHERE LOWER({self.partition_key}) = '{partition.lower()}'" if partition else ""};
                    """

            df = self.from_sqlite(sql=sql)
        else:  # if no sqlite connection supplied, read the data on-the-fly
            df = self.read_dataframe(columns=(self.LANGUAGE_KEY, self.TEXT_KEY))
            if partition:  # on-the-fly equivalent of above `WHERE` clause
                df = df[df[self.partition_key].str.lower() == partition.lower()]
            # on-the-fly equivalent of above `LOWER` & `SUBSTR` functions
            df[self.language_key] = df[self.language_key].str.lower()

        return df

    @cached_property
    def train_dataset(self) -> pd.DataFrame:
        return self.get_base_dataset(partition="train")

    @cached_property
    def test_dataset(self) -> pd.DataFrame:
        return self.get_base_dataset(partition="test")
