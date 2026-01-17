"""Evaluation pipeline for QD-MSA model on ISIC2017 dataset."""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.preprocessing import get_test_transforms
from data.dataset import ISIC2017Dataset

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator class for QD-MSA model."""

    # Class names for ISIC2017
    CLASS_NAMES = ['melanoma', 'seborrheic_keratosis', 'nevus']

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = 'cuda',
    ):
        """Initialize the evaluator.

        Args:
            model: The model to evaluate.
            test_loader: Test data loader.
            device: Device to evaluate on.
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on the test set.

        Returns:
            Dictionary containing evaluation metrics.
        """
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = output.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities)

        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets, probabilities)

        return metrics

    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict[str, Any]:
        """Calculate evaluation metrics.

        Args:
            predictions: Predicted class labels.
            targets: True class labels.
            probabilities: Prediction probabilities.

        Returns:
            Dictionary of metrics.
        """
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision_macro = precision_score(targets, predictions, average='macro', zero_division=0)
        precision_weighted = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall_macro = recall_score(targets, predictions, average='macro', zero_division=0)
        recall_weighted = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(targets, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(targets, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)

        # Confusion matrix
        conf_matrix = confusion_matrix(targets, predictions)

        # Classification report
        class_report = classification_report(
            targets, predictions,
            target_names=self.CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )

        # AUC-ROC (if we have probabilities)
        try:
            if len(np.unique(targets)) > 1:
                auc_macro = roc_auc_score(targets, probabilities, multi_class='ovr', average='macro')
                auc_weighted = roc_auc_score(targets, probabilities, multi_class='ovr', average='weighted')
            else:
                auc_macro = 0.0
                auc_weighted = 0.0
        except ValueError:
            auc_macro = 0.0
            auc_weighted = 0.0

        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'auc_macro': auc_macro,
            'auc_weighted': auc_weighted,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'per_class': {
                self.CLASS_NAMES[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1': float(f1_per_class[i]),
                } for i in range(len(self.CLASS_NAMES)) if i < len(precision_per_class)
            },
        }

        return metrics

    def print_results(self, metrics: Dict[str, Any]):
        """Print evaluation results in a formatted way.

        Args:
            metrics: Dictionary of metrics from evaluate().
        """
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)

        print(f"\nOverall Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']*100:.2f}%")
        print(f"  Precision (macro):  {metrics['precision_macro']*100:.2f}%")
        print(f"  Precision (weighted): {metrics['precision_weighted']*100:.2f}%")
        print(f"  Recall (macro):     {metrics['recall_macro']*100:.2f}%")
        print(f"  Recall (weighted):  {metrics['recall_weighted']*100:.2f}%")
        print(f"  F1 Score (macro):   {metrics['f1_macro']*100:.2f}%")
        print(f"  F1 Score (weighted): {metrics['f1_weighted']*100:.2f}%")
        print(f"  AUC-ROC (macro):    {metrics['auc_macro']*100:.2f}%")
        print(f"  AUC-ROC (weighted): {metrics['auc_weighted']*100:.2f}%")

        print(f"\nPer-Class Metrics:")
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"  {class_name}:")
            print(f"    Precision: {class_metrics['precision']*100:.2f}%")
            print(f"    Recall:    {class_metrics['recall']*100:.2f}%")
            print(f"    F1 Score:  {class_metrics['f1']*100:.2f}%")

        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])

        print("="*60 + "\n")


def load_model(
    checkpoint_path: str,
    model_type: str = 'split',
    num_classes: int = 3,
    hidden_dim: int = 512,
    device: str = 'cuda',
) -> nn.Module:
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model_type: 'split' or 'unsplit'.
        num_classes: Number of output classes.
        hidden_dim: Hidden dimension of the model.
        device: Device to load model to.

    Returns:
        Loaded model.
    """
    from train import create_model

    # Create model
    model = create_model(
        num_classes=num_classes,
        model_type=model_type,
        hidden_dim=hidden_dim,
        device=device,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info("Loaded model from %s (epoch %d, val_acc: %.2f%%)",
                checkpoint_path, checkpoint['epoch'], checkpoint['best_val_acc'])

    return model


def evaluate_model(
    data_root: str,
    checkpoint_path: str,
    model_type: str = 'split',
    batch_size: int = 32,
    device: str = 'cuda',
    num_workers: int = 4,
) -> Dict[str, Any]:
    """Evaluate a trained model on the test set.

    Args:
        data_root: Root directory of the ISIC2017 dataset.
        checkpoint_path: Path to the model checkpoint.
        model_type: 'split' or 'unsplit'.
        batch_size: Batch size.
        device: Device to evaluate on.
        num_workers: Number of data loading workers.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Create test dataset and loader
    test_dataset = ISIC2017Dataset(
        root=data_root,
        split='test',
        transform=get_test_transforms(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load model
    model = load_model(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=device,
    )

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    # Evaluate
    metrics = evaluator.evaluate()
    evaluator.print_results(metrics)

    return metrics


if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='Evaluate QD-MSA model')
    parser.add_argument('--data-root', type=str, required=True, help='Path to ISIC2017 dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='split', choices=['split', 'unsplit'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    metrics = evaluate_model(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        batch_size=args.batch_size,
        device=args.device,
    )
