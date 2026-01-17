"""Dataset classes for ISIC2017 skin cancer dataset."""

from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, List
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class ISIC2017Dataset(Dataset):
    """PyTorch Dataset for ISIC2017 skin cancer dataset.

    The ISIC2017 dataset contains dermoscopic images with three classes:
    - melanoma (malignant)
    - seborrheic_keratosis (benign)
    - nevus (benign)

    Expected directory structure:
    root/
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
    """

    # Class labels mapping
    CLASSES = {
        0: 'melanoma',
        1: 'seborrheic_keratosis',
        2: 'nevus'
    }

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """Initialize ISIC2017 dataset.

        Args:
            root: Root directory of the dataset.
            split: Dataset split ('train', 'val', or 'test').
            transform: Optional transform for images.
            target_transform: Optional transform for labels.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Set paths based on split
        self._setup_paths()

        # Load labels and image paths
        self.samples = self._load_samples()

    def _setup_paths(self):
        """Setup file paths based on split."""
        if self.split == 'train':
            self.data_dir = self.root / 'ISIC-2017_Training_Data'
            self.labels_file = self.root / 'ISIC-2017_Training_Part3_GroundTruth.csv'
        elif self.split == 'val':
            self.data_dir = self.root / 'ISIC-2017_Validation_Data'
            self.labels_file = self.root / 'ISIC-2017_Validation_Part3_GroundTruth.csv'
        elif self.split == 'test':
            self.data_dir = self.root / 'ISIC-2017_Test_v2_Data'
            self.labels_file = self.root / 'ISIC-2017_Test_v2_Part3_GroundTruth.csv'
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'.")

    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load image paths and labels.

        Returns:
            List of (image_path, label) tuples.
        """
        samples = []

        # Check if labels file exists
        if not self.labels_file.exists():
            # Return empty list if dataset not downloaded
            return samples

        # Load labels CSV
        df = pd.read_csv(self.labels_file)

        # ISIC2017 CSV format: image_id, melanoma, seborrheic_keratosis
        # nevus is implicit (neither melanoma nor seborrheic_keratosis)
        for _, row in df.iterrows():
            image_id = row['image_id']
            image_path = self.data_dir / f"{image_id}.jpg"

            if not image_path.exists():
                continue

            # Determine class label
            if row['melanoma'] == 1.0:
                label = 0  # melanoma
            elif row['seborrheic_keratosis'] == 1.0:
                label = 1  # seborrheic_keratosis
            else:
                label = 2  # nevus

            samples.append((image_path, label))

        return samples

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image, label).
        """
        image_path, label = self.samples[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset.

        Returns:
            Dictionary mapping class names to counts.
        """
        counts = {name: 0 for name in self.CLASSES.values()}
        for _, label in self.samples:
            counts[self.CLASSES[label]] += 1
        return counts


def create_data_loaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, validation, and test sets.

    Args:
        root: Root directory of the dataset.
        batch_size: Batch size for data loaders.
        num_workers: Number of worker processes for data loading.
        train_transform: Transform for training data.
        val_transform: Transform for validation data.
        test_transform: Transform for test data.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Create datasets
    train_dataset = ISIC2017Dataset(root, split='train', transform=train_transform)
    val_dataset = ISIC2017Dataset(root, split='val', transform=val_transform)
    test_dataset = ISIC2017Dataset(root, split='test', transform=test_transform)

    # Create data loaders
    # Only use shuffle=True and drop_last=True if dataset is not empty
    train_shuffle = len(train_dataset) > 0
    train_drop_last = len(train_dataset) >= batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train_drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
