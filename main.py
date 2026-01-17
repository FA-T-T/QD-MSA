"""Main entry point for QD-MSA framework.

This script provides a unified CLI for training, evaluation, and inference
with the QD-MSA quantum-classical hybrid model for skin cancer classification.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def cmd_train(args):
    """Execute training command."""
    from train import train_model

    history = train_model(
        data_root=args.data_root,
        model_type=args.model_type,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
    )

    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


def cmd_test(args):
    """Execute test/evaluation command."""
    from test import evaluate_model

    metrics = evaluate_model(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
    )

    print(f"\nTest Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Test F1 (macro): {metrics['f1_macro']*100:.2f}%")


def cmd_info(args):
    """Display dataset and model information."""
    from data.utils import verify_dataset, get_dataset_stats, download_dataset

    if args.data_root:
        print("\n" + "="*60)
        print("DATASET INFORMATION")
        print("="*60)

        if verify_dataset(args.data_root):
            stats = get_dataset_stats(args.data_root)

            for split, split_stats in stats.items():
                print(f"\n{split.upper()} SET:")
                print(f"  Total samples: {split_stats['total']}")
                print(f"  Melanoma: {split_stats['melanoma']}")
                print(f"  Seborrheic Keratosis: {split_stats['seborrheic_keratosis']}")
                print(f"  Nevus: {split_stats['nevus']}")
        else:
            download_dataset(args.data_root)

    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print("\nAvailable model types:")
    print("  - split: Quantum split model (uses n/2+1 qubits)")
    print("  - unsplit: Quantum unsplit model (uses n qubits)")
    print("\nQuantum configuration (split model):")
    print("  - Upper circuit qubits: 4")
    print("  - Lower circuit qubits: 5")
    print("  - Total qubits (split): 5 (n/2 + 1)")
    print("\nQuantum configuration (unsplit model):")
    print("  - Total qubits: 8")
    print("="*60 + "\n")


def cmd_download(args):
    """Download dataset command."""
    from data.utils import download_dataset

    download_dataset(args.data_root)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='QD-MSA: Quantum Distributed Multimodal Sentiment Analysis Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python main.py train --data-root ./data/ISIC2017 --epochs 100

  # Evaluate a trained model
  python main.py test --data-root ./data/ISIC2017 --checkpoint checkpoints/best_model.pth

  # Get dataset info
  python main.py info --data-root ./data/ISIC2017

  # Download dataset (shows instructions)
  python main.py download --data-root ./data/ISIC2017
        """
    )

    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data-root', type=str, required=True, help='Path to ISIC2017 dataset')
    train_parser.add_argument('--model-type', type=str, default='split', choices=['split', 'unsplit'],
                             help='Quantum model type (default: split)')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    train_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    train_parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (default: 1e-4)')
    train_parser.add_argument('--device', type=str, default='cuda', help='Device (default: cuda)')
    train_parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                             help='Checkpoint directory (default: checkpoints)')
    train_parser.add_argument('--num-workers', type=int, default=4, help='Data loader workers (default: 4)')
    train_parser.set_defaults(func=cmd_train)

    # Test command
    test_parser = subparsers.add_parser('test', help='Evaluate a trained model')
    test_parser.add_argument('--data-root', type=str, required=True, help='Path to ISIC2017 dataset')
    test_parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    test_parser.add_argument('--model-type', type=str, default='split', choices=['split', 'unsplit'],
                            help='Quantum model type (default: split)')
    test_parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    test_parser.add_argument('--device', type=str, default='cuda', help='Device (default: cuda)')
    test_parser.add_argument('--num-workers', type=int, default=4, help='Data loader workers (default: 4)')
    test_parser.set_defaults(func=cmd_test)

    # Info command
    info_parser = subparsers.add_parser('info', help='Show dataset and model information')
    info_parser.add_argument('--data-root', type=str, help='Path to ISIC2017 dataset')
    info_parser.set_defaults(func=cmd_info)

    # Download command
    download_parser = subparsers.add_parser('download', help='Download dataset (shows instructions)')
    download_parser.add_argument('--data-root', type=str, required=True, help='Directory to store dataset')
    download_parser.set_defaults(func=cmd_download)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose)

    args.func(args)


if __name__ == '__main__':
    main()
