import logging
from typing import Self

from psycopg import connect as postgres_connect
from psycopg.abc import Params
from rich.logging import RichHandler

from yolo_dataset_db_manager.db.abstract import AbstractDatabaseManager
from yolo_dataset_db_manager.settings import ParamsConnection

FORMAT = "%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

FetchRow = tuple[str, bytes, str, str, str]
QueryParams = FetchRow | Params


class PostgreSQLManager(AbstractDatabaseManager):
    """
    PostgreSQL-specific implementation of AbstractDatabaseManager.

    Handles connections, queries, and table operations for a PostgreSQL database.
    """

    def __init__(
        self,
        params: ParamsConnection,
        table_name: str | None = None,
        create_table: bool = False,
    ) -> None:
        self.params = params.model_dump()
        self.table_name = table_name
        if create_table:
            self.create_table_if_not_exists()

    def db_connection(self) -> Self:
        self.connection = postgres_connect(**self.params)
        self.cursor = self.connection.cursor()
        return self

    def create_table_if_not_exists(self) -> None:
        """
        Creates a table if it doesn't already exist.
        """
        if not self.table_name:
            raise ValueError("Table name is missing.")
        query = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name}
            (
                id SERIAL NOT NULL,
                folder CHARACTER VARYING(128) NOT NULL,
                image BYTEA NOT NULL,
                image_name CHARACTER VARYING(255) NOT NULL,
                image_extension CHARACTER VARYING(10) NOT NULL,
                label_content TEXT NOT NULL,
                CONSTRAINT {self.table_name}_pkey PRIMARY KEY (id)
            )
        """
        with self.db_connection() as db:
            db.cursor.execute(query)
            db.commit()

    def fetch_dataset(self, query: str | None = None) -> list[FetchRow]:
        """
        Retrieves dataset information from the database.
        """
        if not query and not self.table_name:
            raise ValueError("Query or table name is missing.")
        if not query:
            query = f"""
                SELECT folder, image, image_name, image_extension, label_content
                FROM {self.table_name}
            """
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def insert_dataset(
        self,
        query: str,
        params: QueryParams,
    ) -> None:
        """
        Inserts a dataset record into the database.
        """
        if not query and not self.table_name:
            raise ValueError("Query or table name is missing.")
        if not query:
            query = f"""
                INSERT INTO {self.table_name} (folder, image, image_name, image_extension, label_content)
                VALUES (%s, %s, %s, %s, %s)
            """
        self.cursor.execute(query, params)

    def commit(self) -> None:
        """
        Commits the current transaction.
        """
        self.connection.commit()

    def close(self) -> None:
        self.cursor.close()
        self.connection.close()
