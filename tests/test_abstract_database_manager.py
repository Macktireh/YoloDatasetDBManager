import pytest

from yolo_dataset_db_manager.db.abstract import AbstractDatabaseManager


def test_abstract_database_manager_methods():
    """Ensures AbstractDatabaseManager cannot be instantiated directly."""
    with pytest.raises(TypeError):
        AbstractDatabaseManager()
