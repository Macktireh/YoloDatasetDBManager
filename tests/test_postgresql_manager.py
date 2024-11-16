import pytest

from yolo_dataset_db_manager.db.postgresql import PostgreSQLManager
from yolo_dataset_db_manager.settings import ParamsConnection


@pytest.fixture
def db_params():
    """Database connection parameters for testing."""
    return ParamsConnection(
        dbname="test_db",
        user="test_user",
        password="test_pass",
        host="localhost",
        port=5432,
    )


def test_postgresql_manager_initialization(db_params):
    """Tests that PostgreSQLManager initializes correctly."""
    manager = PostgreSQLManager(db_params, table_name="test_table")
    assert manager.params["dbname"] == "test_db"
    assert manager.table_name == "test_table"


def test_postgresql_manager_create_table(mock_db_manager):
    """Tests the create_table_if_not_exists method."""
    mock_db_manager.table_name = "mock_table"
    mock_db_manager.create_table_if_not_exists()
    mock_db_manager.db_connection().cursor.execute.assert_called()


def test_postgresql_manager_fetch_dataset(mock_db_manager):
    """Tests the fetch_dataset method."""
    mock_db_manager.table_name = "mock_table"
    query = "SELECT * FROM mock_table"
    result = mock_db_manager.fetch_dataset(query)
    assert result == []
