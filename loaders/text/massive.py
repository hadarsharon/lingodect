import glob
from functools import cached_property
from pathlib import Path
from typing import Optional, Literal

import pandas as pd

from utils.config import Paths
from utils.datasets import BaseTextDataset


class Massive(BaseTextDataset):
    """
    A loader class pertaining to the MASSIVE dataset.
    Includes helper methods for processing, preparing and transforming the dataset for NLP models used in this app.
    """
    DATA_DIR = Paths.DATASETS / "text/MASSIVE/data"
    DATA_EXT = ".jsonl.gz"
    DATA_FILES = glob.glob(rf"{DATA_DIR}/*{DATA_EXT}")
    LANGUAGES: list[str] = sorted(set(Path(file).name[:2] for file in DATA_FILES))
    COUNTRIES: list[str] = sorted(set(Path(file).stem[-2:] for file in DATA_FILES))
    SQLITE3_COLUMNS = ("locale", "partition", "utt")
    SQLITE3_TABLE = "text_massive"
    LANGUAGE_KEY = "locale"
    TEXT_KEY = "utt"
    PARTITION_KEY = "partition"
    PARTITIONS = ("train", "test", "dev")

    @staticmethod
    def read_dataframe(language: Optional[str] = None, columns: Optional[tuple[str, ...]] = None) -> pd.DataFrame:
        """
        Return the dataset in a pandas DataFrame form.
        Includes support for filtering on specific languages and/or columns from the dataset.
        """
        if language:
            try:  # fetch language-specific file
                [language_file] = filter(lambda f: Path(f).name[:2] == language, Massive.DATA_FILES)
            except ValueError:
                raise FileNotFoundError(
                    f"{Massive.DATA_EXT} data file for language '{language}' not found in {Massive.DATA_DIR}"
                ) from None  # suppress original exception as it's not useful
            else:
                df = pd.read_json(language_file, lines=True)[list(columns)] \
                    if columns else pd.read_json(language_file, lines=True)
        else:  # read all language(s) files
            df = pd.concat(
                (
                    pd.read_json(file, lines=True)[list(columns)] if columns else pd.read_json(file, lines=True)
                    for file in Massive.DATA_FILES
                )
            )
        return df

    def get_base_dataset(self, partition: Optional[Literal["train", "test", "dev"]] = None) -> pd.DataFrame:
        """
        Get the base dataset as-is in pandas DataFrame form, with an optional split by partition
        """
        if self.sqlite_conn:  # prefer sqlite as it's faster than reading on-the-fly
            sql = f"""
                    SELECT 
                        LOWER(SUBSTR({self.language_key}, 1, 2)) AS {self.language_key}, 
                        {self.text_key} AS {self.text_key} 
                    FROM {self.db_table}
                    {f"WHERE LOWER({self.partition_key}) = '{partition.lower()}'" if partition else ""};
                    """

            df = self.from_sqlite(sql=sql)
        else:  # if no sqlite connection supplied, read the data on-the-fly
            df = self.read_dataframe(columns=(self.LANGUAGE_KEY, self.TEXT_KEY, self.PARTITION_KEY))
            if partition:  # on-the-fly equivalent of above `WHERE` clause
                df = df[df[self.partition_key].str.lower() == partition.lower()]
            # on-the-fly equivalent of above `LOWER` & `SUBSTR` functions
            df[self.language_key] = df[self.language_key].str.slice(start=0, stop=2).str.lower()

        return df

    @cached_property
    def train_dataset(self) -> pd.DataFrame:
        return self.get_base_dataset(partition="train")

    @cached_property
    def test_dataset(self) -> pd.DataFrame:
        return self.get_base_dataset(partition="test")

    @cached_property
    def dev_dataset(self) -> pd.DataFrame:
        return self.get_base_dataset(partition="dev")
