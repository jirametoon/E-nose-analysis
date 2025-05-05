import os
import json
import torch
import pickle
import numpy as np
import pandas as pd

from models.cnn import CNN1D
from models.lstm import LSTMNet
from models.transformer import TransformerNet
from .data_utils import preprocess_single_cycle, scale_features, clean_columns

# Define model classes mapping
MODEL_CLASSES = {
    'cnn': CNN1D,
    'lstm': LSTMNet,
    'transformer': TransformerNet
}

def load_model(model_path, metadata_path=None, scaler_path=None):
    """
    Load a trained E-NOSE model with associated metadata.
    
    Args:
        model_path: Path to the PyTorch model file (.pt)
        metadata_path: Path to the model metadata file (.json)
        scaler_path: Path to the scaler file (.pkl) - not used with new preprocessing
        
    Returns:
        dict: A dictionary containing the model, metadata
    """
    # Determine model type from path if metadata not provided
    if metadata_path is None:
        # Try to find metadata in the same directory
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split('.')[0]
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    
    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        # If metadata not found, try to infer from model path
        model_filename = os.path.basename(model_path)
        if 'cnn' in model_filename.lower():
            model_type = 'cnn'
        elif 'lstm' in model_filename.lower():
            model_type = 'lstm'
        elif 'transformer' in model_filename.lower():
            model_type = 'transformer'
        else:
            raise ValueError(f"Could not determine model type from filename: {model_filename}")
        
        metadata = {
            'model_type': model_type,
            'num_classes': 45,  # Default for E-NOSE (45 smells)
            'cycle_length': 60,  # Default cycle length
            'label_map': {}  # Will be empty but should be filled by caller
        }
    
    # Determine model class
    model_type = metadata.get('model_type', 'cnn')  # Default to CNN if not specified
    ModelClass = MODEL_CLASSES.get(model_type)
    
    if ModelClass is None:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Initialize model with correct parameters
    if model_type == 'cnn':
        model = ModelClass(
            input_channels=15,
            seq_length=metadata.get('cycle_length', 60),
            num_classes=metadata.get('num_classes', 45)
        )
    elif model_type == 'lstm':
        model = ModelClass(
            input_size=15,
            num_classes=metadata.get('num_classes', 45)
        )
    elif model_type == 'transformer':
        model = ModelClass(
            input_dim=15,
            num_classes=metadata.get('num_classes', 45)
        )
    
    # Load model weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume the checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return {
        'model': model,
        'metadata': metadata,
        'device': device
    }

def predict_smell(model_data, cycle_data):
    """
    Use a trained model to predict the smell class from cycle data.
    
    Args:
        model_data: Dictionary containing model and metadata
        cycle_data: Numpy array of shape [seq_length, input_features]
                   representing sensor readings for one cycle
        
    Returns:
        dict: A dictionary with prediction results
    """
    model = model_data['model']
    metadata = model_data['metadata']
    device = model_data['device']
    
    # Preprocess the cycle data using the updated preprocessing
    cycle_length = metadata.get('cycle_length', 60)
    preprocessed_data = preprocess_single_cycle(cycle_data, cycle_length)
    
    # Convert to the format expected by the model
    model_type = metadata.get('model_type')
    
    if model_type == 'cnn':
        # CNN expects [batch, channels, sequence]
        input_tensor = torch.tensor(preprocessed_data, dtype=torch.float32)
        input_tensor = input_tensor.transpose(1, 2).to(device)
    else:
        # LSTM and Transformer expect [batch, sequence, features]
        input_tensor = torch.tensor(preprocessed_data, dtype=torch.float32).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item() * 100
    
    # Get label from metadata
    label_map = metadata.get('label_map', {})
    predicted_label = label_map.get(str(predicted_class), f"Class {predicted_class}")
    
    # Get top 3 predictions
    top_k = 3
    top_probs, top_classes = torch.topk(probs, k=min(top_k, outputs.shape[1]))
    
    top_predictions = []
    for i in range(min(top_k, len(top_classes[0]))):
        class_idx = top_classes[0][i].item()
        class_name = label_map.get(str(class_idx), f"Class {class_idx}")
        prob = top_probs[0][i].item() * 100
        top_predictions.append({
            'label': class_name,
            'probability': prob
        })
    
    return {
        'predicted_class': predicted_class,
        'predicted_label': predicted_label,
        'confidence': confidence,
        'top_predictions': top_predictions
    }

def predict_from_dataframe(model_data, df, cycle, sensor_cols=None):
    """
    Make predictions from a DataFrame by extracting a specific cycle.
    
    Args:
        model_data: Dictionary containing model and metadata
        df: Pandas DataFrame with sensor readings
        cycle: Cycle number to use for prediction
        sensor_cols: List of sensor column names
        
    Returns:
        dict: Prediction results
    """
    if sensor_cols is None:
        sensor_cols = [f'G{i}' for i in range(1, 16)]
    
    # First, clean the DataFrame with the same preprocessing pipeline
    df = df.copy()
    
    # Clean column names
    df = clean_columns(df)
    
    # Extract cycle data
    cycle_data = df[df['Cycle Loop'] == cycle][sensor_cols].values
    
    # Apply MinMaxScaler to the cycle data
    cycle_df = df[df['Cycle Loop'] == cycle].copy()
    cycle_df = scale_features(cycle_df)
    cycle_data = cycle_df[sensor_cols].values
    
    # Make prediction
    return predict_smell(model_data, cycle_data)