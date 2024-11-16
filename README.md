# YoloDatasetDBManager

**YoloDatasetDBManager** is a Python library designed to efficiently manage YOLO datasets stored in a PostgreSQL database. It enables saving images and their associated annotations, rebuilding folder structures from the database, and performing flexible operations on datasets.

## 📥 Installation

#### Via `pip`

Ensure Python 3.12+ is installed, then run:

```bash
pip install YoloDatasetDBManager
```

#### Via `pdm`

If you use [PDM](https://pdm-project.org/en/latest/), run:

```bash
pdm add YoloDatasetDBManager
```

## 🚀 Usage

Here’s a complete example of how to use **YoloDatasetDBManager**:

```python
import os
from pathlib import Path

from yolo_dataset_db_manager.db import PostgreSQLManager
from yolo_dataset_db_manager.processor import YoloDatasetProcessor
from yolo_dataset_db_manager.settings import ParamsConnection

if __name__ == "__main__":
    params = ParamsConnection(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
    )
    base_directory = Path(__file__).parent.parent
    dataset_path = base_directory / "data"
    output_path = base_directory / "datasets_out"

    # Initialize the database manager and process datasets
    with PostgreSQLManager(params, table_name="yolo_dataset", create_table=True) as db:
        process = YoloDatasetProcessor(
            db, dataset_path=dataset_path, output_path=output_path
        )
        process.save_dataset()
        process.rebuild_dataset()
        dataset = db.fetch_dataset()
```

## 🛠️ Contribution

Contributions are welcome! Follow these steps to get involved:

1. Fork the project on GitHub.
2. Clone the repository:
   ```bash
   git clone https://github.com/your-username/YoloDatasetDBManager.git
   ```
3. Install development dependencies with `pdm`:
   ```bash
   pdm install -d
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b feature/my-feature
   ```
5. Submit a pull request when your work is ready.

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

Feel free to customize the GitHub link or content as needed! 😊
