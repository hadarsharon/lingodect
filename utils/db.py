"""
Utility module for DB-related operations (inserting/fetching data etc.)
"""

import logging
import sqlite3
import sys
from functools import cached_property
from typing import NoReturn

from utils.config import Paths

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class SQLiteDB:
    """
    Implementation for a SQLite-based Database Backend
    """
    SQLITE_PATH = Paths.DATASETS / "db.sqlite"

    def __init__(self):
        pass

    def get_connection(self) -> sqlite3.Connection | NoReturn:
        logger.info(f"Connecting to SQLite3 Database in {self.SQLITE_PATH}...")
        try:
            conn: sqlite3.Connection = sqlite3.connect(self.SQLITE_PATH)
        except Exception:
            logger.error("Something went wrong while trying to connect to SQLite3 Database, aborting...")
            raise
        else:
            logger.info("Success.")
            return conn

    @cached_property
    def conn(self) -> sqlite3.Connection:
        return self.get_connection()
