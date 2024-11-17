import logging
from pathlib import Path

from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from yolo_dataset_db_manager.db.abstract import AbstractDatabaseManager

FORMAT = "%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

progress_bar = Progress(
    TextColumn("[progress.description]{task.description}"),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)


class YoloDatasetProcessor:
    """
    Handles processing and reconstruction of YOLO datasets from a database.

    Includes saving datasets to a database and rebuilding folder structures.
    """

    def __init__(self, db_manager: AbstractDatabaseManager, dataset_path: Path, output_path: Path) -> None:
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.db_manager = db_manager
        self.folders = [
            "train",
            "valid",
            "test",
        ]

    def save_dataset(self, query: str | None = None) -> None:
        """Saves dataset records into the database."""
        supported_extensions = [".jpg", ".jpeg", ".png"]
        with progress_bar as p:
            for folder in p.track(self.folders, description="Saving dataset..."):
                image_folder = self.dataset_path / folder / "images"
                label_folder = self.dataset_path / folder / "labels"

                for image_path in image_folder.glob("*.*"):
                    if not image_path.is_file():
                        continue

                    if image_path.suffix.lower() not in supported_extensions:
                        logging.warning(f"File ignored: {image_path.name} (extension not supported)")
                        continue
                    try:
                        with open(image_path, "rb") as image_file:
                            image_data = image_file.read()

                        label_path = label_folder / f"{image_path.stem}.txt"
                        if not label_path.exists():
                            logging.warning(f"Label file not found: {label_path.name}")
                            continue

                        with open(label_path) as label_file:
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
                        logging.error(f"Error processing file: {image_path.name}")
                        logging.error(e)

    def rebuild_dataset(self, query: str | None = None) -> None:
        """Rebuilds the YOLO dataset folder structure from the database."""
        dataset = self.db_manager.fetch_dataset(query)

        with progress_bar as p:
            for folder, image_data, image_name, image_extension, label_content in p.track(
                dataset, description="Rebuilding folders"
            ):
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
