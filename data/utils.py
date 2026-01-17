"""Utility functions for ISIC2017 dataset management."""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def download_dataset(root: str, force: bool = False) -> bool:
    """Download the ISIC2017 dataset.

    Note: The ISIC2017 dataset requires registration and agreement to terms.
    This function provides instructions for manual download.

    Args:
        root: Directory to store the dataset.
        force: If True, re-download even if files exist.

    Returns:
        True if dataset is available, False otherwise.
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if not force and verify_dataset(root):
        logger.info("ISIC2017 dataset already exists at %s", root)
        return True

    # Provide download instructions
    download_instructions = """
    ================================================================================
    ISIC2017 Skin Cancer Dataset Download Instructions
    ================================================================================

    The ISIC2017 dataset requires manual download from the ISIC Archive.

    1. Visit the ISIC Challenge website:
       https://challenge.isic-archive.com/data/#2017

    2. Download the following files:
       - ISIC-2017_Training_Data.zip
       - ISIC-2017_Training_Part3_GroundTruth.csv
       - ISIC-2017_Validation_Data.zip
       - ISIC-2017_Validation_Part3_GroundTruth.csv
       - ISIC-2017_Test_v2_Data.zip
       - ISIC-2017_Test_v2_Part3_GroundTruth.csv

    3. Extract the zip files to: {root}

    Expected directory structure:
    {root}/
        ISIC-2017_Training_Data/
            ISIC_0000000.jpg
            ...
        ISIC-2017_Training_Part3_GroundTruth.csv
        ISIC-2017_Validation_Data/
            ...
        ISIC-2017_Validation_Part3_GroundTruth.csv
        ISIC-2017_Test_v2_Data/
            ...
        ISIC-2017_Test_v2_Part3_GroundTruth.csv

    ================================================================================
    """.format(root=root)

    logger.info(download_instructions)
    print(download_instructions)

    return False


def verify_dataset(root: str) -> bool:
    """Verify that the ISIC2017 dataset is properly downloaded and structured.

    Args:
        root: Root directory of the dataset.

    Returns:
        True if dataset is valid, False otherwise.
    """
    root_path = Path(root)

    # Required directories and files
    required_items = [
        # Training
        'ISIC-2017_Training_Data',
        'ISIC-2017_Training_Part3_GroundTruth.csv',
        # Validation
        'ISIC-2017_Validation_Data',
        'ISIC-2017_Validation_Part3_GroundTruth.csv',
        # Test
        'ISIC-2017_Test_v2_Data',
        'ISIC-2017_Test_v2_Part3_GroundTruth.csv',
    ]

    for item in required_items:
        item_path = root_path / item
        if not item_path.exists():
            logger.warning("Missing: %s", item_path)
            return False

    # Check for at least some images in each split
    splits = [
        'ISIC-2017_Training_Data',
        'ISIC-2017_Validation_Data',
        'ISIC-2017_Test_v2_Data',
    ]

    for split_dir in splits:
        split_path = root_path / split_dir
        images = list(split_path.glob('*.jpg'))
        if len(images) == 0:
            logger.warning("No images found in %s", split_path)
            return False

    logger.info("ISIC2017 dataset verified successfully at %s", root)
    return True


def get_dataset_stats(root: str) -> dict:
    """Get statistics about the ISIC2017 dataset.

    Args:
        root: Root directory of the dataset.

    Returns:
        Dictionary with dataset statistics.
    """
    import pandas as pd

    root_path = Path(root)
    stats = {
        'train': {'total': 0, 'melanoma': 0, 'seborrheic_keratosis': 0, 'nevus': 0},
        'val': {'total': 0, 'melanoma': 0, 'seborrheic_keratosis': 0, 'nevus': 0},
        'test': {'total': 0, 'melanoma': 0, 'seborrheic_keratosis': 0, 'nevus': 0},
    }

    label_files = {
        'train': 'ISIC-2017_Training_Part3_GroundTruth.csv',
        'val': 'ISIC-2017_Validation_Part3_GroundTruth.csv',
        'test': 'ISIC-2017_Test_v2_Part3_GroundTruth.csv',
    }

    for split, filename in label_files.items():
        filepath = root_path / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            stats[split]['total'] = len(df)
            stats[split]['melanoma'] = int(df['melanoma'].sum())
            stats[split]['seborrheic_keratosis'] = int(df['seborrheic_keratosis'].sum())
            stats[split]['nevus'] = stats[split]['total'] - stats[split]['melanoma'] - stats[split]['seborrheic_keratosis']

    return stats
