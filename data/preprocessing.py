"""Image preprocessing utilities for ISIC2017 skin cancer dataset."""

from PIL import Image
import torch
from torchvision import transforms


class ISICPreprocessor:
    """Preprocessor for ISIC2017 skin lesion images.

    ISIC2017 dataset contains dermoscopic images of skin lesions
    with three classes: melanoma, seborrheic keratosis, and nevus.
    """

    def __init__(self, image_size: int = 224, normalize: bool = True):
        """Initialize the preprocessor.

        Args:
            image_size: Target size for images (square).
            normalize: Whether to apply ImageNet normalization.
        """
        self.image_size = image_size
        self.normalize = normalize
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a single image.

        Args:
            image: PIL Image to preprocess.

        Returns:
            Preprocessed tensor of shape (3, H, W).
        """
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert to tensor
        tensor = transforms.ToTensor()(image)

        # Normalize
        if self.normalize:
            tensor = transforms.Normalize(self.mean, self.std)(tensor)

        return tensor

    def preprocess_batch(self, images: list) -> torch.Tensor:
        """Preprocess a batch of images.

        Args:
            images: List of PIL Images.

        Returns:
            Batch tensor of shape (N, 3, H, W).
        """
        tensors = [self.preprocess(img) for img in images]
        return torch.stack(tensors, dim=0)


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Get training data augmentation transforms.

    Args:
        image_size: Target image size.

    Returns:
        Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Get validation data transforms (no augmentation).

    Args:
        image_size: Target image size.

    Returns:
        Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_test_transforms(image_size: int = 224) -> transforms.Compose:
    """Get test data transforms (no augmentation).

    Args:
        image_size: Target image size.

    Returns:
        Composed transform pipeline.
    """
    return get_val_transforms(image_size)
