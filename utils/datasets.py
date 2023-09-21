"""
Module for external Dataset-related integrations & configurations
Datasets are used for fitting and training the language detection models
"""
import logging
import sqlite3
import sys
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional, NoReturn, Literal

import pandas as pd
from keras.layers import StringLookup
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset

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
        """
        Helper method to support loading an external data to a SQLite Database
        Each dataset may override this method for more specific implementations or other intricate cases
        """
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
        """
        Helper method to support loading a dataset persisted on an SQLite Database, to a pandas DataFrame
        Each dataset may override this method for more specific implementations or other intricate cases
        """
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

    @property
    @abstractmethod
    def train_dataset(self) -> pd.DataFrame:
        """
        Should return the partition in the dataset for training the model
        For example: as part of fitting the parameters to the model when it's learning
        """
    @property
    @abstractmethod
    def test_dataset(self) -> pd.DataFrame:
        """
        Should return the partition in the dataset for testing the model
        For example: to test the accuracy of the model immediately after having trained it
        """
    @property
    @abstractmethod
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
    IMAGE_HEIGHT: int
    IMAGE_WIDTH: int

    def __init__(
            self,
            image_path_key: str,
            image_transcription_key: str
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

    def prepare_dataset(self, partition: Literal["train", "test", "dev"]) -> PrefetchDataset:
        """
        This method should prepare the dataset to be forwarded to the model
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
