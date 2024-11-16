from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from yolo_dataset_db_manager.db import PostgreSQLManager
from yolo_dataset_db_manager.settings import ParamsConnection


@pytest.fixture
def temp_dir():
    """Temporary directory for dataset paths."""
    with TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_db_manager():
    """Mock instance of PostgreSQLManager."""
    db_params = ParamsConnection(
        dbname="test_db",
        user="test_user",
        password="test_pass",
        host="localhost",
        port=5432,
    )
    db_manager = PostgreSQLManager(db_params)
    db_manager.db_connection = MagicMock(return_value=db_manager)

    # Mock cursor and connection
    db_manager.cursor = MagicMock()
    db_manager.connection = MagicMock()
    db_manager.cursor.execute = MagicMock()
    db_manager.cursor.fetchall = MagicMock(return_value=[])

    db_manager.fetch_dataset = MagicMock(return_value=[])
    db_manager.insert_dataset = MagicMock()
    db_manager.commit = MagicMock()
    db_manager.close = MagicMock()
    yield db_manager
