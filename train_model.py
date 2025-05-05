#!/usr/bin/env python
"""
E-Nose Model Training Script

This script trains and evaluates deep learning models for electronic nose (E-Nose)
smell classification. It supports CNN, LSTM, and Transformer architectures.

Usage:
    python train_model.py --model cnn --data_dir data/15cycle_datasets --epochs 200

Author: E-Nose Team
Date: May 2025
"""

import os
import argparse
import time
import json
import pickle
from typing import Dict, Tuple, List, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from models.cnn import CNN1D
from models.lstm import LSTMNet
from models.transformer import TransformerNet
from utils.data_utils import load_smell_dataset, create_dataloaders
from utils.viz_utils import plot_metrics, plot_confusion_matrix

# Constants
MODEL_CLASSES = {
    'cnn': CNN1D,
    'lstm': LSTMNet,
    'transformer': TransformerNet
}

# Type aliases for clarity
ModelType = Union[CNN1D, LSTMNet, TransformerNet]
DeviceType = torch.device
History = Dict[str, List[float]]
LabelMap = Dict[int, str]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the training script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Train E-NOSE classification models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='cnn', 
        choices=MODEL_CLASSES.keys(),
        help='Model architecture to use'
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='data/15cycle_datasets',
        help='Directory containing smell CSV files'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=200,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--cycle_length', 
        type=int, 
        default=60,
        help='Number of rows per cycle'
    )
    parser.add_argument(
        '--num_workers', 
        type=int, 
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training (cuda or cpu)'
    )
    parser.add_argument(
        '--save_interval', 
        type=int, 
        default=50,
        help='Save model checkpoint every N epochs'
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='',
        help='Custom model name prefix (default: model architecture name)'
    )
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For completely deterministic results, may impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_device(device_str: str) -> Tuple[DeviceType, bool]:
    """Set up and validate the training device (CPU/GPU).
    
    Args:
        device_str: Device string ('cpu', 'cuda', or 'cuda:n')
        
    Returns:
        tuple: (device, is_cuda_available)
    """
    print("\n===== GPU DEBUGGING INFO =====")
    print(f"PyTorch version: {torch.__version__}")
    
    is_cuda_available = torch.cuda.is_available()
    
    if is_cuda_available:
        # Force CUDA usage - ignore any previous device setting
        cuda_device = 0
        device_str = f'cuda:{cuda_device}'
        torch.cuda.set_device(cuda_device)
        
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"CUDA Device: {torch.cuda.get_device_name(cuda_device)}")
        print(f"CUDA Device Index: {torch.cuda.current_device()}")
        
        # Test GPU memory allocation to confirm it's working
        try:
            test_tensor = torch.tensor([1.0], device=device_str)
            print(f"GPU Test: Successful - tensor on {test_tensor.device}")
            del test_tensor  # Free up memory
            torch.cuda.empty_cache()  # Clear CUDA cache
            
            # Enable for better performance with variable input sizes
            torch.backends.cudnn.benchmark = True
        except Exception as e:
            print(f"GPU Test Failed with error: {str(e)}")
            print("Falling back to CPU...")
            device_str = 'cpu'
            is_cuda_available = False
    else:
        print("WARNING: CUDA is not available! Training will use CPU.")
        print("Install PyTorch with CUDA support to use your GPU.")
        device_str = 'cpu'
    
    print(f"Training will use: {device_str.upper()}")
    print("=============================\n")
    
    return torch.device(device_str), is_cuda_available


def format_time(seconds: float) -> str:
    """Format time in seconds to hours, minutes, seconds.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"


def train_epoch(
    model: ModelType, 
    loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    device: DeviceType
) -> Tuple[float, float]:
    """Train the model for one epoch.
    
    Args:
        model: The neural network model
        loader: Training data loader
        optimizer: Optimization algorithm
        criterion: Loss function
        device: Device to use for training
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    # Calculate average statistics
    avg_loss = train_loss / len(loader.dataset)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(
    model: ModelType, 
    loader: DataLoader, 
    criterion: nn.Module, 
    device: DeviceType
) -> Tuple[float, float]:
    """Validate the model on the validation set.
    
    Args:
        model: The neural network model
        loader: Validation data loader
        criterion: Loss function
        device: Device to use for validation
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Track statistics
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    # Calculate average statistics
    avg_loss = val_loss / len(loader.dataset)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def save_checkpoint(
    model: ModelType, 
    optimizer: torch.optim.Optimizer,
    epoch: int,
    accuracy: float,
    history: History,
    path: str,
    is_best: bool = False
) -> None:
    """Save model checkpoint.
    
    Args:
        model: The neural network model
        optimizer: Optimization algorithm
        epoch: Current epoch
        accuracy: Validation accuracy
        history: Training history
        path: Path to save the checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': accuracy,
        'history': history
    }
    
    if is_best:
        print(f"Saving best model with accuracy {accuracy:.2f}%...")
    else:
        print(f"Saving checkpoint at epoch {epoch+1}...")
        
    torch.save(checkpoint, path)
    
    # Clear CUDA cache after saving
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_model(
    model: ModelType, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    scheduler: Any, 
    criterion: nn.Module, 
    device: DeviceType, 
    epochs: int, 
    save_dir: str, 
    save_interval: int = 50, 
    model_name: str = ''
) -> Tuple[History, float]:
    """Train the model.
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        test_loader: Validation data loader
        optimizer: Optimization algorithm
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device to use for training
        epochs: Number of epochs to train
        save_dir: Directory to save model checkpoints
        save_interval: Save checkpoints every N epochs
        model_name: Custom name for model checkpoints
        
    Returns:
        tuple: (training_history, best_accuracy)
    """
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    save_threshold = 1.0  # Only save when accuracy improves by at least this much
    last_saved_epoch = -1
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get base model name for checkpoints
    if not model_name:
        model_name = 'model'
    
    # Best model path
    best_model_path = os.path.join(save_dir, f"{model_name}_best.pt")
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Models will be saved every {save_interval} epochs")
    print(f"Best model will be saved as: {best_model_path}")
    
    # Print GPU info if available
    if torch.cuda.is_available() and str(device).startswith('cuda'):
        print(f"\nGPU Memory Stats Before Training:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Cached:    {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation phase
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print statistics
        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")
        
        # Print estimated remaining time
        remaining_epochs = epochs - (epoch + 1)
        remaining_time = epoch_time * remaining_epochs
        print(f"Estimated time remaining: {format_time(remaining_time)}")
        
        # Save best model
        if val_acc > best_acc:
            improvement = val_acc - best_acc
            # Only save if significant improvement or certain epochs passed
            if improvement >= save_threshold or (epoch - last_saved_epoch >= 100):
                save_checkpoint(
                    model, optimizer, epoch, val_acc, history,
                    best_model_path, is_best=True
                )
                last_saved_epoch = epoch
                print(f"New best accuracy: {val_acc:.2f}% (improved by {improvement:.2f}%)")
            
            # Always update best_acc even if we don't save
            best_acc = val_acc
        
        # Save checkpoint at specified intervals
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"{model_name}_{epoch+1}.pt")
            save_checkpoint(
                model, optimizer, epoch, val_acc, history,
                checkpoint_path, is_best=False
            )
    
    print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
    
    return history, best_acc


def evaluate_model(
    model: ModelType, 
    test_loader: DataLoader, 
    criterion: nn.Module, 
    device: DeviceType, 
    label_map: LabelMap
) -> Tuple[float, float, np.ndarray, List[int], List[int]]:
    """Evaluate the trained model.
    
    Args:
        model: The neural network model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to use for evaluation
        label_map: Mapping from class indices to class names
        
    Returns:
        tuple: (test_loss, test_accuracy, confusion_matrix, predictions, targets)
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Track statistics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Store predictions and targets for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate average test statistics
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = 100.0 * correct / total
    
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Get classification report
    class_names = [label_map[i] for i in range(len(label_map))]
    report = classification_report(all_targets, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    return test_loss, test_acc, cm, all_preds, all_targets


def save_model_artifacts(
    model_dir: str,
    model_name: str,
    model_type: str,
    version: str,
    history: History,
    label_map: LabelMap,
    cycle_length: int,
    best_acc: float,
    test_acc: float,
    test_loss: float,
    cm: np.ndarray,
    scaler: Any
) -> None:
    """Save model artifacts including plots, metadata, and scaler.
    
    Args:
        model_dir: Directory to save artifacts
        model_name: Model name for filenames
        model_type: Model architecture type
        version: Model version
        history: Training history
        label_map: Mapping from class indices to class names
        cycle_length: Number of rows per cycle
        best_acc: Best validation accuracy
        test_acc: Test accuracy
        test_loss: Test loss
        cm: Confusion matrix
        scaler: Feature scaler used for preprocessing
    """
    # Plot metrics with standardized naming
    metrics_save_path = os.path.join(model_dir, f"{model_name}_metrics.png")
    plot_metrics(
        history, 
        save_path=metrics_save_path, 
        model_name=model_type, 
        version=version
    )
    
    # Plot confusion matrix with standardized naming
    class_names = [label_map[i] for i in range(len(label_map))]
    cm_save_path = os.path.join(model_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(
        cm, 
        class_names, 
        save_path=cm_save_path,
        model_name=model_type, 
        version=version
    )
    
    # Save model metadata
    metadata = {
        'model_type': model_type,
        'version': version,
        'num_classes': len(label_map),
        'label_map': label_map,
        'cycle_length': cycle_length,
        'best_accuracy': float(best_acc),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Also save scaler for preprocessing future data
    scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Model artifacts saved to {model_dir}:")
    print(f"  - Metrics plot: {metrics_save_path}")
    print(f"  - Confusion matrix: {cm_save_path}")
    print(f"  - Metadata: {metadata_path}")
    print(f"  - Scaler: {scaler_path}")


def optimize_hyperparameters(args: argparse.Namespace, is_cuda: bool) -> argparse.Namespace:
    """Optimize hyperparameters based on device and other settings.
    
    Args:
        args: Command line arguments
        is_cuda: Whether CUDA is available
        
    Returns:
        argparse.Namespace: Optimized arguments
    """
    # Make a copy of args to avoid modifying the original
    optimized_args = args
    
    if is_cuda:
        # For GPU training, larger batch sizes are more efficient
        optimized_args.batch_size = max(64, args.batch_size)
        print(f"Optimized batch size for GPU: {optimized_args.batch_size}")
        
        # Reduce number of workers for better GPU utilization
        optimized_args.num_workers = min(2, args.num_workers)  
        print(f"Optimized worker count for GPU: {optimized_args.num_workers}")
    
    return optimized_args


def create_model(model_type: str, num_classes: int, cycle_length: int) -> ModelType:
    """Create model instance based on model type.
    
    Args:
        model_type: Type of model to create ('cnn', 'lstm', or 'transformer')
        num_classes: Number of output classes
        cycle_length: Number of rows per cycle
        
    Returns:
        Model instance
    """
    ModelClass = MODEL_CLASSES[model_type]
    
    if model_type == 'cnn':
        model = ModelClass(
            input_channels=15,  # 15 gas sensors (G1-G15)
            seq_length=cycle_length,
            num_classes=num_classes
        )
    elif model_type == 'lstm':
        model = ModelClass(
            input_size=15,  # 15 gas sensors (G1-G15)
            num_classes=num_classes
        )
    elif model_type == 'transformer':
        model = ModelClass(
            input_dim=15,  # 15 gas sensors (G1-G15)
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    return model


def main() -> None:
    """Main function for training and evaluation."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up device (CPU/GPU)
    device, is_cuda = setup_device(args.device)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Optimize hyperparameters based on device
    args = optimize_hyperparameters(args, is_cuda)
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset_info = load_smell_dataset(args.data_dir, args.cycle_length, model_type=args.model)
    
    train_dataset = dataset_info['train_dataset']
    test_dataset = dataset_info['test_dataset']
    label_map = dataset_info['label_map']
    num_classes = dataset_info['num_classes']
    scaler = dataset_info['scaler']
    
    # Create data loaders
    loaders = create_dataloaders(train_dataset, test_dataset, args.batch_size, args.num_workers)
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    
    print(f"Dataset loaded with {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    print(f"Found {num_classes} smell classes: {list(label_map.values())}")
    
    # Initialize model
    print(f"Initializing {args.model.upper()} model...")
    model = create_model(args.model, num_classes, args.cycle_length)
    
    # Move model to device
    model = model.to(device)
    print(f"Model initialized on {device}")
    
    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Set up model directory structure and naming
    save_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(save_dir, exist_ok=True)
    
    # Set model name
    model_name = args.model_name if args.model_name else args.model
    
    # Full path to best model (this is where train_model will save the best model)
    best_model_path = os.path.join(save_dir, f"{model_name}_best.pt")
    
    # Train model
    print(f"Starting training process...")
    history, best_acc = train_model(
        model, train_loader, test_loader, optimizer, scheduler, criterion,
        device, args.epochs, save_dir, 
        save_interval=args.save_interval,
        model_name=model_name
    )
    
    # Evaluate the best model
    print("Loading best model for evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Evaluating model on test set...")
    test_loss, test_acc, cm, all_preds, all_targets = evaluate_model(
        model, test_loader, criterion, device, label_map
    )
    
    # Extract version from model name (e.g., "cnn_v1" -> "v1")
    if "_" in model_name:
        model_type, version = model_name.split("_", 1)
    else:
        model_type = args.model
        version = "v1"  # Default version if not specified
    
    # Save model artifacts
    save_model_artifacts(
        save_dir,
        model_name,
        model_type,
        version,
        history,
        label_map,
        args.cycle_length,
        best_acc,
        test_acc,
        test_loss,
        cm,
        scaler
    )
    
    print(f"Training completed. Models and metadata saved to {save_dir}")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"\nSaved models:")
    print(f"  - Best model: {best_model_path}")
    
    # Only print periodic checkpoints if save_interval > 0
    if args.save_interval > 0:
        for epoch in range(args.save_interval, args.epochs + 1, args.save_interval):
            print(f"  - Epoch {epoch}: {os.path.join(save_dir, f'{model_name}_{epoch}.pt')}")


if __name__ == "__main__":
    main()