voici la nouvelle structure est ce que on améliorer et refactoriser ? et ajoute les docstrings

── yolo_dataset_manager/
  │   ├── __init__.py
  │   ├── db/
  │   │   ├── abstract.py
  │   │   └── postgresql.py
  │   ├── processor.py        
  │   ├── settings.py       
...

```py
# yolo_dataset_manager/db/abstract.py

from abc import ABC, abstractmethod
from typing import Self

from psycopg.abc import Params


FetchRow = tuple[str, bytes, str, str, str]
QueryParams = FetchRow | Params


class AbstractDatabaseManager(ABC):
    """
    Manages PostgreSQL connections and queries.
    """

    def __enter__(self) -> Self:
        return self.db_connection()

    @abstractmethod
    def db_connection(self) -> Self:
        pass

    @abstractmethod
    def create_table_if_not_exists(self) -> None:
        """
        Creates a table if it doesn't already exist.
        """
        pass

    @abstractmethod
    def fetch_dataset(self, query: str) -> list[FetchRow]:
        """
        Retrieves dataset information from the database.
        """
        pass

    @abstractmethod
    def insert_dataset(
        self,
        query: str,
        params: QueryParams,
    ) -> None:
        """
        Inserts a dataset record into the database.
        """
        pass

    @abstractmethod
    def commit(self) -> None:
        """
        Commits the current transaction.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
```
```py
# yolo_dataset_manager/db/postgresql.py

from typing import Self

from psycopg import connect as postgres_connect
from psycopg.abc import Params

from yolo_dataset_manager.db.abstract import AbstractDatabaseManager
from yolo_dataset_manager.settings import ParamsConnection

FetchRow = tuple[str, bytes, str, str, str]
QueryParams = FetchRow | Params


class PostgreSQLManager(AbstractDatabaseManager):
    """
    Manages PostgreSQL connections and queries.
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
```

```py
# yolo_dataset_manager/processor.py

import logging
from pathlib import Path

from yolo_dataset_manager.db.abstract import AbstractDatabaseManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class YoloDatasetProcessor:
    def __init__(
        self, db_manager: AbstractDatabaseManager, dataset_path: Path, output_path: Path
    ) -> None:
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.db_manager = db_manager
        self.folders = [
            "train",
            "valid",
            "test",
        ]

    def save_dataset(self, query: str | None = None) -> None:
        supported_extensions = [".jpg", ".jpeg", ".png"]
        for folder in self.folders:
            image_folder = self.dataset_path / folder / "images"
            label_folder = self.dataset_path / folder / "labels"

            for image_path in image_folder.glob("*.*"):
                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in supported_extensions:
                    logging.warning(
                        f"Fichier ignoré : {image_path.name} (extension non prise en charge)"
                    )
                    continue

                try:
                    with open(image_path, "rb") as image_file:
                        image_data = image_file.read()

                    label_path = label_folder / f"{image_path.stem}.txt"
                    if not label_path.exists():
                        logging.warning(f"Label introuvable pour {image_path.name}")
                        continue

                    with open(label_path, "r") as label_file:
                        label_content = label_file.read()

                    query_params = (
                        folder,
                        image_data,
                        image_path.stem,
                        image_path.suffix,
                        label_content,
                    )
                    self.db_manager.insert_dataset(query=query, params=query_params)
                except Exception as e:
                    logging.error(
                        f"Erreur lors du traitement de {image_path.name}: {e}"
                    )

    def rebuild_dataset(self, query: str | None = None) -> None:
        dataset = self.db_manager.fetch_dataset(query)

        for folder, image_data, image_name, image_extension, label_content in dataset:
            # Paths for images and labels
            image_folder = self.output_path / folder / "images"
            label_folder = self.output_path / folder / "labels"
            image_folder.mkdir(parents=True, exist_ok=True)
            label_folder.mkdir(parents=True, exist_ok=True)

            image_path = image_folder / f"{image_name}{image_extension}"
            with open(image_path, "wb") as image_file:
                image_file.write(image_data)

            label_path = label_folder / f"{image_name}.txt"
            with open(label_path, "w") as label_file:
                label_file.write(label_content)
```

```py
# yolo_dataset_manager/settings.py

from pydantic import BaseModel, Field, field_validator


class ParamsConnection(BaseModel):
    """Encapsulates and validates PostgreSQL connection parameters."""

    dbname: str = Field(description="Database name.")
    user: str = Field(description="Login username.")
    password: str = Field(description="User password.")
    host: str = Field(description="PostgreSQL server address.")
    port: int = Field(default=5432, description="PostgreSQL server port.")

    @field_validator("dbname", "user", "password", "host")
    def validate_non_empty(cls, value, field) -> str:
        if not value:
            raise ValueError(f"`{field.name}` cannot be empty.")
        return value

    @field_validator("port")
    def validate_port(cls, value) -> str:
        if not isinstance(value, int):
            raise ValueError("`port` must be an integer.")
        return value
```