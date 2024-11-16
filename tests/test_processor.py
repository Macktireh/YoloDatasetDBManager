from yolo_dataset_db_manager.processor import YoloDatasetProcessor


def test_processor_initialization(mock_db_manager, temp_dir):
    """Tests initialization of YoloDatasetProcessor."""
    processor = YoloDatasetProcessor(
        db_manager=mock_db_manager,
        dataset_path=temp_dir,
        output_path=temp_dir / "output",
    )
    assert processor.dataset_path == temp_dir
    assert processor.output_path == temp_dir / "output"
    assert processor.db_manager == mock_db_manager


def test_save_dataset(mock_db_manager, temp_dir):
    """Tests the save_dataset method."""
    processor = YoloDatasetProcessor(
        db_manager=mock_db_manager,
        dataset_path=temp_dir,
        output_path=temp_dir / "output",
    )

    # Simulate files in the dataset directory
    train_images_dir = temp_dir / "train/images"
    train_labels_dir = temp_dir / "train/labels"
    train_images_dir.mkdir(parents=True)
    train_labels_dir.mkdir(parents=True)

    image_path = train_images_dir / "image1.jpg"
    label_path = train_labels_dir / "image1.txt"

    image_path.write_bytes(b"fake_image_data")
    label_path.write_text("label_content")

    processor.save_dataset()

    # Ensure insert_dataset is called
    mock_db_manager.insert_dataset.assert_called()


def test_rebuild_dataset(mock_db_manager, temp_dir):
    """Tests the rebuild_dataset method."""
    mock_db_manager.fetch_dataset.return_value = [
        ("train", b"fake_image_data", "image1", ".jpg", "label_content"),
    ]
    processor = YoloDatasetProcessor(
        db_manager=mock_db_manager,
        dataset_path=temp_dir,
        output_path=temp_dir / "output",
    )

    processor.rebuild_dataset()

    # Check that the file was written to the correct location
    output_image = temp_dir / "output/train/images/image1.jpg"
    output_label = temp_dir / "output/train/labels/image1.txt"

    assert output_image.exists()
    assert output_label.exists()
    assert output_image.read_bytes() == b"fake_image_data"
    assert output_label.read_text() == "label_content"
