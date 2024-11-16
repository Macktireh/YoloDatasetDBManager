from abc import ABC, abstractmethod
from typing import Self

from psycopg.abc import Params

FetchRow = tuple[str, bytes, str, str, str]
QueryParams = FetchRow | Params


class AbstractDatabaseManager(ABC):
    """
    Abstract base class for managing database connections and queries.

    Subclasses must implement methods for connecting to the database,
    executing queries, and handling transactions.
    """

    def __enter__(self) -> Self:
        """Context manager entry point to set up the database connection."""
        return self.db_connection()

    @abstractmethod
    def db_connection(self) -> Self:
        """Establish a database connection."""
        pass

    @abstractmethod
    def create_table_if_not_exists(self) -> None:
        """Create a table if it does not exist."""
        pass

    @abstractmethod
    def fetch_dataset(self, query: str) -> list[FetchRow]:
        """Fetch dataset records based on a query."""
        pass

    @abstractmethod
    def insert_dataset(self, query: str, params: QueryParams) -> None:
        """Insert dataset records into the database."""
        pass

    @abstractmethod
    def commit(self) -> None:
        """Commit the current transaction."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point to close the database connection."""
        self.close()
