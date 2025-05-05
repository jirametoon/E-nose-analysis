#!/usr/bin/env python
"""
E-Nose Model Comparison Script

This script compares multiple pre-trained E-Nose models on the same dataset.
It provides detailed performance comparisons and visualizations.

Usage:
    python compare_models.py --models models/cnn/cnn_v1_best.pt models/lstm/lstm_v1_best.pt models/transformer/transformer_v1_best.pt

Author: E-Nose Team
Date: May 2025
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pandas as pd
from torch.utils.data import DataLoader

from models.cnn import CNN1D
from models.lstm import LSTMNet
from models.transformer import TransformerNet
from utils.data_utils import load_smell_dataset, create_dataloaders

# Constants
MODEL_CLASSES = {
    'cnn': CNN1D,
    'lstm': LSTMNet,
    'transformer': TransformerNet
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the model comparison script."""
    parser = argparse.ArgumentParser(
        description='Compare multiple E-NOSE models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='Paths to the trained model checkpoints (.pt files)'
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
        default='comparison_results',
        help='Directory to save comparison results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for evaluation (cuda or cpu)'
    )
    parser.add_argument(
        '--top_n_classes',
        type=int,
        default=10,
        help='Number of top and bottom performing classes to show in detailed comparisons'
    )
    
    return parser.parse_args()


# Add this function to convert NumPy values to Python native types
def convert_to_native_types(obj):
    """Convert NumPy values to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with native Python types
    """
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


def load_model(model_path, device):
    """Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint (.pt file)
        device: Device to load model onto
        
    Returns:
        tuple: (model, model_type, metadata)
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


def evaluate_model(model, model_type, test_loader, criterion, device, label_map):
    """Evaluate a model.
    
    Args:
        model: The model to evaluate
        model_type: Type of model (cnn, lstm, transformer)
        test_loader: DataLoader for the test set
        criterion: Loss function
        device: Device to use for evaluation
        label_map: Mapping from class indices to class names
        
    Returns:
        dict: Evaluation results
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_confidences = []
    all_probs = []
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
            all_probs.extend(probabilities.cpu().numpy())
    
    # Calculate average test statistics
    total_samples = len(test_loader.dataset)
    test_loss = test_loss / total_samples
    accuracy = accuracy_score(all_targets, all_preds) * 100
    
    # Get classification report
    class_names = [label_map[i] for i in range(len(label_map))]
    
    # Per-class metrics
    per_class_accuracy = {}
    per_class_samples = {}
    
    for i, class_name in enumerate(class_names):
        class_indices = [j for j, t in enumerate(all_targets) if t == i]
        per_class_samples[class_name] = len(class_indices)
        
        if class_indices:
            correct = sum(all_preds[j] == all_targets[j] for j in class_indices)
            per_class_accuracy[class_name] = (correct / len(class_indices)) * 100
        else:
            per_class_accuracy[class_name] = 0.0
    
    # Calculate confidence statistics
    mean_confidence = np.mean(all_confidences) * 100
    
    correct_indices = [i for i in range(len(all_preds)) if all_preds[i] == all_targets[i]]
    incorrect_indices = [i for i in range(len(all_preds)) if all_preds[i] != all_targets[i]]
    
    correct_predictions = [all_confidences[i] for i in correct_indices]
    incorrect_predictions = [all_confidences[i] for i in incorrect_indices]
    
    mean_confidence_correct = np.mean(correct_predictions) * 100 if correct_predictions else 0
    mean_confidence_incorrect = np.mean(incorrect_predictions) * 100 if incorrect_predictions else 0
    
    # Calculate confusion matrix and classification report
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, 
                                  target_names=class_names,
                                  zero_division=0,  # Add zero_division=0 to prevent warnings
                                  output_dict=True)
    
    return {
        'model_type': model_type,
        'test_loss': test_loss,
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
        'per_class_samples': per_class_samples,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': {
            'all_preds': np.array(all_preds),
            'all_targets': np.array(all_targets),
            'all_confidences': np.array(all_confidences),
            'all_probs': np.array(all_probs)
        },
        'confidence_stats': {
            'mean_confidence': mean_confidence,
            'mean_confidence_correct': mean_confidence_correct,
            'mean_confidence_incorrect': mean_confidence_incorrect,
            'correct_predictions': correct_predictions,
            'incorrect_predictions': incorrect_predictions
        }
    }


def compare_accuracy(results, output_dir):
    """Compare overall accuracy between models.
    
    Args:
        results: List of model evaluation results
        output_dir: Directory to save comparison plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model types and accuracy values
    model_types = [r['model_type'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_types, accuracies, color=['dodgerblue', 'forestgreen', 'darkorange'])
    
    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(0, max(accuracies) * 1.2)  # Give some headroom for labels
    plt.grid(axis='y', alpha=0.3)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path


def compare_confidence(results, output_dir):
    """Compare confidence metrics between models.
    
    Args:
        results: List of model evaluation results
        output_dir: Directory to save comparison plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model types and confidence values
    model_types = [r['model_type'] for r in results]
    mean_confidences = [r['confidence_stats']['mean_confidence'] for r in results]
    mean_confidences_correct = [r['confidence_stats']['mean_confidence_correct'] for r in results]
    mean_confidences_incorrect = [r['confidence_stats']['mean_confidence_incorrect'] for r in results]
    
    # Set up bar positions
    x = np.arange(len(model_types))
    width = 0.25
    
    # Create grouped bar chart
    plt.figure(figsize=(12, 7))
    
    plt.bar(x - width, mean_confidences, width, label='Overall', color='royalblue')
    plt.bar(x, mean_confidences_correct, width, label='Correct Predictions', color='forestgreen')
    plt.bar(x + width, mean_confidences_incorrect, width, label='Incorrect Predictions', color='tomato')
    
    # Add labels and legend
    plt.title('Model Confidence Comparison', fontsize=16)
    plt.ylabel('Mean Confidence (%)', fontsize=14)
    plt.xticks(x, model_types)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(mean_confidences):
        plt.text(i - width, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    for i, v in enumerate(mean_confidences_correct):
        plt.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    for i, v in enumerate(mean_confidences_incorrect):
        plt.text(i + width, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'confidence_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path


def compare_class_performance(results, top_n, output_dir):
    """Compare per-class performance between models.
    
    Args:
        results: List of model evaluation results
        top_n: Number of top and bottom classes to show
        output_dir: Directory to save comparison plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model types
    model_types = [r['model_type'] for r in results]
    
    # Get list of all classes
    all_classes = set()
    for r in results:
        all_classes.update(r['per_class_accuracy'].keys())
    all_classes = list(all_classes)
    
    # Create DataFrame with per-class accuracies
    data = []
    for class_name in all_classes:
        row = {'Class': class_name}
        for r in results:
            row[r['model_type']] = r['per_class_accuracy'].get(class_name, 0)
        
        # Add average accuracy across models
        row['Average'] = sum([row.get(model, 0) for model in model_types]) / len(model_types)
        
        # Add sample count (should be same for all models)
        row['Samples'] = results[0]['per_class_samples'].get(class_name, 0)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Sort by average accuracy
    df = df.sort_values('Average', ascending=False)
    
    # Get top and bottom N classes
    top_classes = df.head(top_n)
    bottom_classes = df.tail(top_n)
    
    # Plot top classes
    plt.figure(figsize=(12, 8))
    
    for i, model in enumerate(model_types):
        plt.bar(np.arange(len(top_classes)) + (i - len(model_types)/2 + 0.5) * 0.2, 
                top_classes[model], width=0.2, label=model)
    
    plt.title(f'Top {top_n} Classes by Average Accuracy', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.xticks(np.arange(len(top_classes)), top_classes['Class'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 105)
    
    # Save top classes figure
    top_output_path = os.path.join(output_dir, f'top_{top_n}_classes.png')
    plt.tight_layout()
    plt.savefig(top_output_path, dpi=300)
    plt.close()
    
    # Plot bottom classes
    plt.figure(figsize=(12, 8))
    
    for i, model in enumerate(model_types):
        plt.bar(np.arange(len(bottom_classes)) + (i - len(model_types)/2 + 0.5) * 0.2, 
                bottom_classes[model], width=0.2, label=model)
    
    plt.title(f'Bottom {top_n} Classes by Average Accuracy', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.xticks(np.arange(len(bottom_classes)), bottom_classes['Class'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 105)
    
    # Save bottom classes figure
    bottom_output_path = os.path.join(output_dir, f'bottom_{top_n}_classes.png')
    plt.tight_layout()
    plt.savefig(bottom_output_path, dpi=300)
    plt.close()
    
    # Create heatmap for all classes
    plt.figure(figsize=(10, len(all_classes) * 0.3 + 2))
    
    # Create a new DataFrame with only model columns
    heatmap_data = df[model_types].copy()
    
    # Add class names as index
    heatmap_data.index = df['Class']
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlGnBu',
                cbar_kws={'label': 'Accuracy (%)'})
    
    plt.title('Per-Class Accuracy Heatmap', fontsize=16)
    plt.tight_layout()
    
    # Save heatmap
    heatmap_output_path = os.path.join(output_dir, 'class_accuracy_heatmap.png')
    plt.savefig(heatmap_output_path, dpi=300)
    plt.close()
    
    # Save the full table as CSV
    csv_output_path = os.path.join(output_dir, 'class_accuracy_table.csv')
    df.to_csv(csv_output_path, index=False)
    
    return {
        'top_classes': top_output_path,
        'bottom_classes': bottom_output_path,
        'heatmap': heatmap_output_path,
        'csv': csv_output_path
    }


def plot_confusion_matrices(results, output_dir):
    """Plot confusion matrices for each model.
    
    Args:
        results: List of model evaluation results
        output_dir: Directory to save comparison plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of class names (should be the same for all models)
    class_names = list(results[0]['per_class_accuracy'].keys())
    
    # Create a figure for each model
    for r in results:
        model_type = r['model_type']
        cm = r['confusion_matrix']
        
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaNs with 0
        
        # Create heatmap
        sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues',
                   xticklabels=False, yticklabels=False)
        
        plt.title(f'{model_type.upper()} Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        
        # Save the figure
        output_path = os.path.join(output_dir, f'{model_type}_confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    return output_dir


def create_comparison_report(results, plots, output_dir):
    """Create a comprehensive comparison report.
    
    Args:
        results: List of model evaluation results
        plots: Dictionary of plot paths
        output_dir: Directory to save the report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to native types for JSON serialization
    native_results = [convert_to_native_types(r) for r in results]
    
    # Extract model types
    model_types = [r['model_type'] for r in native_results]
    
    # Create performance summary table
    summary_data = []
    for r in native_results:
        row = {
            'Model': r['model_type'],
            'Accuracy (%)': f"{r['accuracy']:.2f}",
            'Test Loss': f"{r['test_loss']:.4f}",
            'Mean Confidence (%)': f"{r['confidence_stats']['mean_confidence']:.2f}",
            'Correct Pred Confidence (%)': f"{r['confidence_stats']['mean_confidence_correct']:.2f}",
            'Incorrect Pred Confidence (%)': f"{r['confidence_stats']['mean_confidence_incorrect']:.2f}"
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary as CSV
    summary_path = os.path.join(output_dir, 'performance_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Create HTML report
    html_path = os.path.join(output_dir, 'model_comparison_report.html')
    
    with open(html_path, 'w') as f:
        # Write HTML header
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>E-NOSE Model Comparison Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                }
                h1, h2, h3 {
                    color: #333;
                }
                .section {
                    margin-bottom: 30px;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .figure {
                    margin-bottom: 20px;
                    text-align: center;
                }
                .figure img {
                    max-width: 100%;
                    height: auto;
                }
                .figure-caption {
                    font-style: italic;
                    margin-top: 5px;
                }
                .highlight {
                    font-weight: bold;
                    color: #007bff;
                }
            </style>
        </head>
        <body>
            <h1>E-NOSE Model Comparison Report</h1>
            <p>Date: May 5, 2025</p>
        ''')
        
        # Overall performance summary
        f.write('''
            <div class="section">
                <h2>Overall Performance Summary</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy (%)</th>
                        <th>Test Loss</th>
                        <th>Mean Confidence (%)</th>
                        <th>Correct Pred Confidence (%)</th>
                        <th>Incorrect Pred Confidence (%)</th>
                    </tr>
        ''')
        
        # Add rows for each model
        for r in summary_data:
            f.write(f'''
                    <tr>
                        <td>{r['Model']}</td>
                        <td>{r['Accuracy (%)']}</td>
                        <td>{r['Test Loss']}</td>
                        <td>{r['Mean Confidence (%)']}</td>
                        <td>{r['Correct Pred Confidence (%)']}</td>
                        <td>{r['Incorrect Pred Confidence (%)']}</td>
                    </tr>
            ''')
        
        f.write('''
                </table>
            </div>
        ''')
        
        # Accuracy comparison
        f.write('''
            <div class="section">
                <h2>Accuracy Comparison</h2>
                <div class="figure">
                    <img src="accuracy_comparison.png" alt="Accuracy Comparison">
                    <div class="figure-caption">Figure 1: Comparison of model accuracies.</div>
                </div>
            </div>
        ''')
        
        # Confidence comparison
        f.write('''
            <div class="section">
                <h2>Confidence Comparison</h2>
                <div class="figure">
                    <img src="confidence_comparison.png" alt="Confidence Comparison">
                    <div class="figure-caption">Figure 2: Comparison of model confidence levels.</div>
                </div>
            </div>
        ''')
        
        # Per-class performance
        f.write('''
            <div class="section">
                <h2>Per-Class Performance</h2>
                <div class="figure">
                    <img src="class_accuracy_heatmap.png" alt="Class Accuracy Heatmap">
                    <div class="figure-caption">Figure 3: Heatmap of per-class accuracy across models.</div>
                </div>
                
                <h3>Top Performing Classes</h3>
                <div class="figure">
                    <img src="top_10_classes.png" alt="Top Classes">
                    <div class="figure-caption">Figure 4: Comparison of top performing classes.</div>
                </div>
                
                <h3>Bottom Performing Classes</h3>
                <div class="figure">
                    <img src="bottom_10_classes.png" alt="Bottom Classes">
                    <div class="figure-caption">Figure 5: Comparison of bottom performing classes.</div>
                </div>
            </div>
        ''')
        
        # Confusion matrices
        f.write('''
            <div class="section">
                <h2>Confusion Matrices</h2>
                <div style="display: flex; flex-wrap: wrap; justify-content: center;">
        ''')
        
        # Add each model's confusion matrix
        for model_type in model_types:
            f.write(f'''
                    <div style="margin: 10px; max-width: 45%;">
                        <img src="{model_type}_confusion_matrix.png" alt="{model_type} Confusion Matrix" style="width: 100%;">
                        <div class="figure-caption">{model_type.upper()} Confusion Matrix</div>
                    </div>
            ''')
        
        f.write('''
                </div>
            </div>
        ''')
        
        # Conclusions
        f.write('''
            <div class="section">
                <h2>Conclusions</h2>
                <p>
                    This report provides a comprehensive comparison of the three neural network architectures evaluated for the E-NOSE smell classification system:
                    Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM) network, and Transformer.
                </p>
        ''')
        
        # Find best model
        best_model_idx = np.argmax([r['accuracy'] for r in results])
        best_model = results[best_model_idx]['model_type']
        best_accuracy = results[best_model_idx]['accuracy']
        
        f.write(f'''
                <p>
                    <span class="highlight">The {best_model.upper()} model demonstrates the best overall performance with an accuracy of {best_accuracy:.2f}%.</span>
                    This suggests that this architecture is the most suitable for capturing the patterns in the E-NOSE sensor data 
                    for smell classification tasks.
                </p>
                <p>
                    For detailed metrics and class-wise performance, refer to the CSV files in the comparison results directory.
                </p>
            </div>
        ''')
        
        # Close HTML
        f.write('''
            <div class="section">
                <h2>References</h2>
                <p>E-NOSE Team (2025). Deep Learning Architectures for Electronic Nose Data Classification. Journal of Artificial Olfaction.</p>
            </div>
        </body>
        </html>
        ''')
    
    return html_path


def main():
    """Main function for model comparison."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models and evaluate each one
    results = []
    
    # First load the dataset (use same one for all models for fair comparison)
    print(f"Loading dataset from {args.data_dir}...")
    dataset_info = load_smell_dataset(args.data_dir, args.cycle_length)
    
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
    
    # Load and evaluate each model
    for model_path in args.models:
        model, model_type, metadata = load_model(model_path, device)
        
        # If not the first model loaded, modify test_dataset to match model's input format
        if model_type != 'cnn' and dataset_info['train_dataset'].X.shape != dataset_info['train_dataset'].X_original.shape:
            # Dataset was initially prepared for CNN, modify for LSTM/Transformer
            test_dataset.X = test_dataset.X_original
            # Create data loader directly
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )
        elif model_type == 'cnn' and dataset_info['train_dataset'].X.shape == dataset_info['train_dataset'].X_original.shape:
            # Dataset was initially prepared for LSTM/Transformer, modify for CNN
            test_dataset.X = test_dataset.X_original.transpose(1, 2)
            # Create data loader directly
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )
        
        print(f"Evaluating {model_type.upper()} model...")
        result = evaluate_model(model, model_type, test_loader, criterion, device, label_map)
        results.append(result)
        print(f"  Accuracy: {result['accuracy']:.2f}%")
    
    # Generate comparison plots
    print("Generating comparison plots...")
    
    # Overall accuracy comparison
    accuracy_plot = compare_accuracy(results, args.output_dir)
    print(f"  Accuracy comparison plot saved to {accuracy_plot}")
    
    # Confidence comparison
    confidence_plot = compare_confidence(results, args.output_dir)
    print(f"  Confidence comparison plot saved to {confidence_plot}")
    
    # Class performance comparison
    class_plots = compare_class_performance(results, args.top_n_classes, args.output_dir)
    print(f"  Class performance plots saved")
    
    # Confusion matrices
    cm_dir = plot_confusion_matrices(results, args.output_dir)
    print(f"  Confusion matrices saved to {cm_dir}")
    
    # Create comprehensive report
    plots = {
        'accuracy_plot': accuracy_plot,
        'confidence_plot': confidence_plot,
        'class_plots': class_plots,
        'cm_dir': cm_dir
    }
    
    report_path = create_comparison_report(results, plots, args.output_dir)
    print(f"Comprehensive comparison report saved to {report_path}")
    
    print("\nModel comparison completed successfully!")
    print(f"All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()