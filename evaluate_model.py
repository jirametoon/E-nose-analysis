#!/usr/bin/env python
"""
E-Nose Model Evaluation Script

This script evaluates a pre-trained E-Nose model on a dataset.
It provides detailed performance metrics and visualizations.

Usage:
    python evaluate_model.py --model_path models/cnn/cnn_v1_best.pt --data_dir data/15cycle_datasets

Author: E-Nose Team
Date: May 2025
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from models.cnn import CNN1D
from models.lstm import LSTMNet
from models.transformer import TransformerNet
from utils.data_utils import load_smell_dataset
from utils.viz_utils import plot_confusion_matrix

# Constants
MODEL_CLASSES = {
    'cnn': CNN1D,
    'lstm': LSTMNet,
    'transformer': TransformerNet
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(
        description='Evaluate E-NOSE classification models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/15cycle_datasets',
        help='Directory containing smell CSV files'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
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
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for evaluation (cuda or cpu)'
    )
    
    return parser.parse_args()


def load_model(model_path, device):
    """Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint (.pt file)
        device: Device to load model onto
        
    Returns:
        tuple: (model, model_metadata)
    """
    # Extract model type from path (cnn, lstm, or transformer)
    path_parts = model_path.split('/')
    if len(path_parts) > 1:
        model_type = path_parts[-2]  # models/cnn/model.pt -> cnn
    else:
        # Try to infer from filename
        filename = os.path.basename(model_path)
        if filename.startswith('cnn'):
            model_type = 'cnn'
        elif filename.startswith('lstm'):
            model_type = 'lstm'
        elif filename.startswith('transformer'):
            model_type = 'transformer'
        else:
            raise ValueError(f"Cannot determine model type from path: {model_path}")
    
    print(f"Loading {model_type.upper()} model from {model_path}")
    
    # Look for metadata file in the same directory
    model_dir = os.path.dirname(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # If model name has epoch number, remove it to get base name (e.g., cnn_v1_100.pt -> cnn_v1)
    if '_' in model_name and model_name.split('_')[-1].isdigit():
        base_name = '_'.join(model_name.split('_')[:-1])
    else:
        base_name = model_name
    
    # Try to load metadata
    metadata_path = os.path.join(model_dir, f"{base_name}_metadata.json")
    metadata = None
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded model metadata from {metadata_path}")
    else:
        print(f"No metadata file found at {metadata_path}")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model instance
    if metadata and 'num_classes' in metadata:
        num_classes = metadata['num_classes']
        cycle_length = metadata.get('cycle_length', 60)
    else:
        # Default values if metadata is not available
        print("Warning: No metadata found, using default values")
        num_classes = 45  # Default for our dataset
        cycle_length = 60
    
    # Create appropriate model based on type
    if model_type == 'cnn':
        model = CNN1D(input_channels=15, seq_length=cycle_length, num_classes=num_classes)
    elif model_type == 'lstm':
        model = LSTMNet(input_size=15, num_classes=num_classes)
    elif model_type == 'transformer':
        model = TransformerNet(input_dim=15, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, model_type, metadata


def evaluate_model(model, test_loader, criterion, device, label_map):
    """Evaluate the model on the test set.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for the test set
        criterion: Loss function
        device: Device to use for evaluation
        label_map: Mapping from class indices to class names
        
    Returns:
        dict: Dictionary of evaluation results
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_confidences = []
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Get predictions and confidence scores
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
            
            # Track statistics
            test_loss += loss.item() * inputs.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
    
    # Calculate average test statistics
    total_samples = len(test_loader.dataset)
    test_loss = test_loss / total_samples
    accuracy = accuracy_score(all_targets, all_preds) * 100
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Get classification report with zero_division=0 to suppress warnings
    class_names = [label_map[i] for i in range(len(label_map))]
    report = classification_report(all_targets, all_preds, 
                                  target_names=class_names,
                                  zero_division=0,
                                  output_dict=True)
    
    # Additional metrics
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_indices = [j for j, t in enumerate(all_targets) if t == i]
        if class_indices:
            correct = sum(all_preds[j] == all_targets[j] for j in class_indices)
            per_class_accuracy[class_name] = (correct / len(class_indices)) * 100
    
    # Calculate confidence statistics
    mean_confidence = np.mean(all_confidences) * 100
    correct_predictions = [all_confidences[i] for i in range(len(all_preds)) 
                          if all_preds[i] == all_targets[i]]
    incorrect_predictions = [all_confidences[i] for i in range(len(all_preds)) 
                            if all_preds[i] != all_targets[i]]
    
    mean_confidence_correct = np.mean(correct_predictions) * 100 if correct_predictions else 0
    mean_confidence_incorrect = np.mean(incorrect_predictions) * 100 if incorrect_predictions else 0
    
    # Create results dictionary
    results = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'per_class_accuracy': per_class_accuracy,
        'confidence_stats': {
            'mean_confidence': mean_confidence,
            'mean_confidence_correct': mean_confidence_correct,
            'mean_confidence_incorrect': mean_confidence_incorrect
        },
        'predictions': {
            'all_preds': all_preds,
            'all_targets': all_targets,
            'all_confidences': all_confidences
        }
    }
    
    return results


def plot_class_accuracy(per_class_accuracy, output_path):
    """Plot per-class accuracy.
    
    Args:
        per_class_accuracy: Dictionary mapping class names to accuracy values
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Sort classes by accuracy for better visualization
    sorted_classes = sorted(per_class_accuracy.items(), key=lambda x: x[1], reverse=True)
    classes = [c[0] for c in sorted_classes]
    accuracies = [c[1] for c in sorted_classes]
    
    # Create horizontal bar chart
    bars = plt.barh(classes, accuracies, color='skyblue')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                ha='left', va='center', fontsize=8)
    
    plt.xlabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    plt.xlim(0, 105)  # Limit to 0-100% with a little extra for labels
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_histogram(correct_confidences, incorrect_confidences, output_path):
    """Plot histogram of prediction confidences.
    
    Args:
        correct_confidences: List of confidence values for correct predictions
        incorrect_confidences: List of confidence values for incorrect predictions
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    bins = np.linspace(0, 1, 20)
    
    plt.hist(correct_confidences, bins=bins, alpha=0.7, label='Correct Predictions',
             color='green', density=True)
    plt.hist(incorrect_confidences, bins=bins, alpha=0.7, label='Incorrect Predictions',
             color='red', density=True)
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Distribution of Model Confidence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_results(results, model_type, output_dir):
    """Save evaluation results.
    
    Args:
        results: Dictionary of evaluation results
        model_type: Type of model evaluated
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for cleaner reports
    cm = results['confusion_matrix']
    report = results['classification_report']
    per_class_accuracy = results['per_class_accuracy']
    confidence_stats = results['confidence_stats']
    
    # Prepare predictions data
    all_preds = results['predictions']['all_preds']
    all_targets = results['predictions']['all_targets']
    all_confidences = results['predictions']['all_confidences']
    
    # Prepare data for confidence histogram
    correct_indices = [i for i in range(len(all_preds)) if all_preds[i] == all_targets[i]]
    incorrect_indices = [i for i in range(len(all_preds)) if all_preds[i] != all_targets[i]]
    
    correct_confidences = [float(all_confidences[i]) for i in correct_indices]
    incorrect_confidences = [float(all_confidences[i]) for i in incorrect_indices]
    
    # Convert NumPy values to Python native types for JSON serialization
    def convert_to_native_types(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_types(i) for i in obj]
        else:
            return obj
    
    # Save summary results
    summary = {
        'model_type': model_type,
        'accuracy': float(results['accuracy']),
        'test_loss': float(results['test_loss']),
        'confidence_stats': {
            'mean_confidence': float(confidence_stats['mean_confidence']),
            'mean_confidence_correct': float(confidence_stats['mean_confidence_correct']),
            'mean_confidence_incorrect': float(confidence_stats['mean_confidence_incorrect'])
        },
        'class_wise_metrics': {
            'precision': {c: float(report[c]['precision']) for c in report if c not in ['accuracy', 'macro avg', 'weighted avg']},
            'recall': {c: float(report[c]['recall']) for c in report if c not in ['accuracy', 'macro avg', 'weighted avg']},
            'f1-score': {c: float(report[c]['f1-score']) for c in report if c not in ['accuracy', 'macro avg', 'weighted avg']}
        },
        'macro_avg': convert_to_native_types(report['macro avg']),
        'weighted_avg': convert_to_native_types(report['weighted avg'])
    }
    
    # Save summary as JSON
    with open(os.path.join(output_dir, f'{model_type}_evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save full classification report
    with open(os.path.join(output_dir, f'{model_type}_classification_report.txt'), 'w') as f:
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.2f}%\n")
        f.write(f"Test Loss: {results['test_loss']:.4f}\n\n")
        f.write("Classification Report:\n")
        # Convert classification report dict back to formatted text
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                if label in ['macro avg', 'weighted avg']:
                    f.write(f"\n{label}:\n")
                else:
                    f.write(f"\n{label}:\n")
                for metric_name, value in metrics.items():
                    if isinstance(value, (float, np.floating)):
                        f.write(f"  {metric_name}: {float(value):.4f}\n")
                    else:
                        f.write(f"  {metric_name}: {value}\n")
        
        f.write("\nConfidence Statistics:\n")
        for stat_name, value in confidence_stats.items():
            f.write(f"  {stat_name}: {float(value):.2f}%\n")
    
    # Save confusion matrix plot
    cm_output_path = os.path.join(output_dir, f'{model_type}_confusion_matrix.png')
    label_names = list(per_class_accuracy.keys())
    plot_confusion_matrix(cm, label_names, save_path=cm_output_path,
                         model_name=model_type, version="evaluation")
    
    # Save per-class accuracy plot
    accuracy_output_path = os.path.join(output_dir, f'{model_type}_class_accuracy.png')
    plot_class_accuracy(per_class_accuracy, accuracy_output_path)
    
    # Save confidence histogram
    confidence_output_path = os.path.join(output_dir, f'{model_type}_confidence_histogram.png')
    plot_confidence_histogram(correct_confidences, incorrect_confidences, confidence_output_path)
    
    print(f"Results saved to {output_dir}")
    print(f"  - Summary: {model_type}_evaluation_summary.json")
    print(f"  - Full report: {model_type}_classification_report.txt")
    print(f"  - Confusion matrix: {model_type}_confusion_matrix.png")
    print(f"  - Class accuracy: {model_type}_class_accuracy.png")
    print(f"  - Confidence histogram: {model_type}_confidence_histogram.png")


def main():
    """Main function for model evaluation."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model, model_type, metadata = load_model(args.model_path, device)
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset_info = load_smell_dataset(args.data_dir, args.cycle_length, model_type=model_type)
    
    test_dataset = dataset_info['test_dataset']
    label_map = dataset_info['label_map']
    
    # Create data loader - FIX: Create only the test_loader directly
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"Dataset loaded with {len(test_dataset)} test samples")
    print(f"Found {len(label_map)} smell classes")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, criterion, device, label_map)
    
    # Print summary results
    print("\n===== Evaluation Results =====")
    print(f"Model Type: {model_type}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Mean Confidence: {results['confidence_stats']['mean_confidence']:.2f}%")
    print(f"Mean Confidence (Correct): {results['confidence_stats']['mean_confidence_correct']:.2f}%")
    print(f"Mean Confidence (Incorrect): {results['confidence_stats']['mean_confidence_incorrect']:.2f}%")
    print("=============================\n")
    
    # Save detailed results
    output_dir = os.path.join(args.output_dir, model_type)
    save_results(results, model_type, output_dir)
    
    print(f"Evaluation completed successfully.")


if __name__ == "__main__":
    main()