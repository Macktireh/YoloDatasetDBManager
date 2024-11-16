import logging
from pathlib import Path

from yolo_dataset_db_manager.db.abstract import AbstractDatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
        for folder in self.folders:
            image_folder = self.dataset_path / folder / "images"
            label_folder = self.dataset_path / folder / "labels"

            for image_path in image_folder.glob("*.*"):
                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in supported_extensions:
                    logging.warning(f"Fichier ignoré : {image_path.name} (extension non prise en charge)")
                    continue
                try:
                    with open(image_path, "rb") as image_file:
                        image_data = image_file.read()

                    label_path = label_folder / f"{image_path.stem}.txt"
                    if not label_path.exists():
                        logging.warning(f"Label introuvable pour {image_path.name}")
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
                    logging.error(f"Erreur lors du traitement de {image_path.name}: {e}")

    def rebuild_dataset(self, query: str | None = None) -> None:
        """Rebuilds the YOLO dataset folder structure from the database."""
        dataset = self.db_manager.fetch_dataset(query)

        for folder, image_data, image_name, image_extension, label_content in dataset:
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
