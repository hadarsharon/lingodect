"""
Module for external Dataset-related integrations & configurations
Datasets are used for fitting and training the language detection models
"""
import glob
import logging
import sqlite3
import sys
from abc import ABC
from functools import cached_property
from pathlib import Path
from typing import Optional, NoReturn, Literal

import pandas as pd
from keras.layers import StringLookup

from utils.config import Paths

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class BaseTextDataset(ABC):
    """
    An extensible abstract class for defining a Textual Dataset
    Can later be instantiated and plugged into a detection model
    """
    LANGUAGE_KEY: str
    TEXT_KEY: str
    PARTITION_KEY: str
    SQLITE3_TABLE: str
    SQLITE3_COLUMNS: str

    def __init__(
            self,
            sqlite_conn: Optional[sqlite3.Connection] = None,
            db_table: Optional[str] = None,
            language_key: Optional[str] = None,
            text_key: Optional[str] = None,
            partition_key: Optional[str] = None,
            initialize_db_table: bool = False):
        self.sqlite_conn = sqlite_conn
        self.db_table = db_table or self.SQLITE3_TABLE
        self.language_key = language_key or self.LANGUAGE_KEY
        self.text_key = text_key or self.TEXT_KEY
        self.partition_key = partition_key or self.PARTITION_KEY
        if initialize_db_table:
            assert self.sqlite_conn, "To initialize DB tables, `sqlite_conn` must be passed."
            self.to_sqlite(table=db_table)

    def to_sqlite(
            self,
            table: Optional[str] = None,
            columns: Optional[tuple[str, ...]] = None
    ) -> Optional[NoReturn]:
        if not self.sqlite_conn:
            raise NotImplementedError(
                "'to_sqlite' only supported if class was instantiated with 'sqlite_conn' argument supplied."
            )
        table = table or self.SQLITE3_TABLE
        try:
            df = self.read_dataframe(columns=columns or self.SQLITE3_COLUMNS)
        except Exception:
            logger.error("Something went wrong while trying to read language dataframe, aborting...")
            self.sqlite_conn.close()
            raise
        else:
            logger.info("Done.")
        logger.info(rf"Loading language dataframe(s) to target table: '{table}' in SQLite3 Database...")
        try:
            df.to_sql(table, self.sqlite_conn, if_exists="replace", index=False)
        except Exception:
            logger.error(
                rf"Something went wrong while trying to load language dataframe(s) to target table: '{table}'..."
            )
            raise
        else:
            logger.info("Done.")

    def from_sqlite(
            self,
            sql: Optional[str] = None,
            table: Optional[str] = None,
            columns: Optional[tuple[str, ...]] = None
    ) -> pd.DataFrame:
        if not self.sqlite_conn:
            raise NotImplementedError(
                "'from_sqlite' only supported if class was instantiated with 'sqlite_conn' argument supplied."
            )
        return pd.read_sql(
            sql=sql or rf"SELECT {', '.join(columns) or '*'} FROM {table or self.SQLITE3_TABLE};",
            con=self.sqlite_conn
        )

    @staticmethod
    def read_dataframe(language: Optional[str] = None, columns: Optional[tuple[str, ...]] = None) -> pd.DataFrame:
        """
        Should return a dataframe representation of the raw dataset,
        with an option to filter by language or select specific columns.
        """

    def get_base_dataset(self, partition: Optional[Literal["train", "test", "dev"]] = None) -> pd.DataFrame:
        """
        Should return the base dataset, on which the model will be fitted, trained or tested.
        Optionally, can return a specific partition from it - this can be useful for using them as cached properties.
        """

    def train_dataset(self) -> pd.DataFrame:
        """
        Should return the partition in the dataset for training the model
        For example: as part of fitting the parameters to the model when it's learning
        """

    def test_dataset(self) -> pd.DataFrame:
        """
        Should return the partition in the dataset for testing the model
        For example: to test the accuracy of the model immediately after having trained it
        """

    def dev_dataset(self) -> pd.DataFrame:
        """
        Should return the partition in the dataset for validating the model
        For example: during active development, as part of tuning hyperparameters
        """


class BaseImageDataset(ABC):
    """
    An extensible abstract class for defining an Image-based Dataset
    Can later be instantiated and plugged into a detection model
    """
    IMAGE_PATH_KEY: str
    IMAGE_TRANSCRIPTION_KEY: str

    def __init__(
            self,
            image_path_key: Optional[str] = None,
            image_transcription_key: Optional[str] = None
    ):
        self.image_path_key = image_path_key or self.IMAGE_PATH_KEY
        self.image_transcription_key = image_transcription_key or self.IMAGE_TRANSCRIPTION_KEY

    @staticmethod
    def read_dataframe() -> pd.DataFrame:
        """
        Should return a dataframe representation of the image dataset transcription
        """

    def get_labels(self, partition: Literal["train", "test", "dev"]) -> list[str]:
        """
        Should return the labels (i.e. transcriptions) corresponding to the dataset images in the partition
        """

    def get_vocabulary(self, partition: Literal["train", "test", "dev"]) -> set[str]:
        """
        Should return the vocabulary that comprises the image labels (i.e. transcriptions) in the partition
        """

    @cached_property
    def char_to_num(self) -> StringLookup:
        """
        This method maps characters present in the training vocabulary to integers
        """
        return StringLookup(vocabulary=list(self.get_vocabulary(partition="train")), mask_token=None)

    @cached_property
    def num_to_char(self) -> StringLookup:
        """
        This method maps integers back to the original characters
        """
        return StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(),
            mask_token=None,
            invert=True
        )


class Massive(BaseTextDataset):  # TODO: perhaps take from Hugging Face
    DATA_DIR = Paths.DATASETS / "text/MASSIVE/data"
    DATA_EXT = ".jsonl"
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
            df = self.read_dataframe(columns=(self.LANGUAGE_KEY, self.TEXT_KEY))
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


class Oscar(BaseTextDataset):
    ACCESS_TOKEN = "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"
