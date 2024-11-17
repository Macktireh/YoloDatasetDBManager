# YoloDatasetDBManager

**YoloDatasetDBManager** is a Python library designed to efficiently manage YOLO datasets stored in a PostgreSQL database. It enables saving images and their associated annotations, rebuilding folder structures from the database, and performing flexible operations on datasets.

## 📥 Installation

Ensure Python 3.11+ is installed.

#### Via `pip`

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

# Define the base directory for the project
BASE_DIR = Path(__file__).parent.parent

# Specify the path to the dataset folder, where the YOLO data is stored
dataset_path = BASE_DIR / "data"

# Specify the output directory where the reconstructed dataset will be saved
output_path = BASE_DIR / "datasets_out"

# Configure the connection parameters for the PostgreSQL database
params = ParamsConnection(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
)

# Initialize the PostgreSQL manager and create the YOLO dataset table if it doesn't exist
with PostgreSQLManager(params, table_name="yolo_dataset", create_table=True) as db:
    process = YoloDatasetProcessor(
        db, dataset_path=dataset_path, output_path=output_path
    )

    # Save the dataset (images and annotations) from the local folder to the database
    process.save_dataset()
    db.commit()  # Commit the transaction to ensure data is stored in the database

    # Rebuild the dataset structure from the database into the output folder
    process.rebuild_dataset()
```

## 🛠️ Contribution

Contributions are welcome! Follow these steps to get involved:

1. Fork the project on GitHub.
2. Clone the repository:
   ```bash
   git clone https://github.com/Macktireh/YoloDatasetDBManager.git
   ```
3. Install development dependencies with `pdm`:
   ```bash
   pdm install
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b feature/my-feature
   ```
5. Run tests:
   ```bash
   pdm run test
   ```
6. Submit a pull request when your work is ready.

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
