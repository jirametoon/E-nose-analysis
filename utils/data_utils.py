import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Constants for required columns
REQUIRED_G_COLS = [f'G{i}' for i in range(1, 16)]
CYCLE_COL = 'Cycle Loop'

def validate_structure(df: pd.DataFrame):
    """
    Check if DataFrame contains all required G columns.
    Handles both formats: 'G1' and ' G1 ' (with spaces)
    Returns (True, []) if valid; otherwise, (False, missing_columns).
    """
    # Clean column names for comparison
    df_cols_clean = [col.strip() for col in df.columns]
    missing = [col for col in REQUIRED_G_COLS if col not in df_cols_clean]
    if missing:
        return False, missing
    return True, []

def analyze_cycles(df: pd.DataFrame):
    """
    Return two dictionaries of cycle lengths:
    - too_short: cycles with fewer than 60 rows
    - too_long: cycles with more than 60 rows
    """
    if CYCLE_COL not in df.columns:
        raise ValueError(f"Column '{CYCLE_COL}' not found.")
    grouped = df.groupby(CYCLE_COL)
    too_short = {cycle_id: len(group) for cycle_id, group in grouped if len(group) < 60}
    too_long = {cycle_id: len(group) for cycle_id, group in grouped if len(group) > 60}
    return too_short, too_long

def drop_cycles(df: pd.DataFrame, cycle_ids: list):
    """
    Drop all rows belonging to the specified cycle IDs.
    """
    return df[~df[CYCLE_COL].isin(cycle_ids)]

def trim_cycles(df: pd.DataFrame):
    """
    Trim each cycle to a maximum of 60 rows.
    """
    trimmed = []
    for _, group in df.groupby(CYCLE_COL):
        if len(group) > 60:
            trimmed.append(group.iloc[:60])
        else:
            trimmed.append(group)
    return pd.concat(trimmed, ignore_index=True)

def clean_columns(df: pd.DataFrame):
    """
    Strip leading and trailing whitespace from column names.
    """
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df

def scale_features(df: pd.DataFrame):
    """
    Scale feature columns G1 through G15 to the range [0, 1].
    """
    df = df.copy()
    g_cols = [f'G{i}' for i in range(1, 16)]
    scaler = MinMaxScaler()
    for cycle_id, group in df.groupby(CYCLE_COL.strip()):
        idx = group.index
        df.loc[idx, g_cols] = scaler.fit_transform(group[g_cols])
    return df

class ENoseDataset(Dataset):
    """Dataset for Electronic Nose (E-NOSE) data.
    
    Handles loading and preprocessing of sensor data for PyTorch models.
    """
    
    def __init__(self, X, y=None, model_type='cnn'):
        """
        Initialize the dataset.
        
        Args:
            X: Features (sensor readings), shape [n_samples, seq_length, n_features]
            y: Labels (smell classes), shape [n_samples]
            model_type: Type of model ('cnn' or 'lstm') to determine correct tensor format
        """
        # Store original data as tensor
        self.X_original = torch.tensor(X, dtype=torch.float32)
        
        # For CNN models, transpose from [n_samples, seq_length, n_features] to [n_samples, n_features, seq_length]
        # For LSTM models, keep as [n_samples, seq_length, n_features]
        if model_type == 'cnn':
            self.X = self.X_original.transpose(1, 2)
        else:
            self.X = self.X_original
            
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = None
            
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.X)
    
    def __getitem__(self, idx):
        """Get a sample by index."""
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


def load_smell_dataset(data_dir, cycle_length=60, test_size=0.2, random_state=42, model_type='cnn'):
    """
    Load and prepare the E-NOSE dataset from multiple CSV files.
    Uses the same preprocessing approach as pipeline_streamlit.py.
    
    Args:
        data_dir: Directory containing smell CSV files
        cycle_length: Length of each cycle/segment in rows
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        model_type: Type of model ('cnn' or 'lstm') to determine tensor format
        
    Returns:
        Dictionary with train and test DataLoader objects, label mapping
    """
    all_features = []
    all_labels = []
    label_names = []
    
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for label_idx, csv_file in enumerate(sorted(csv_files)):
        # Extract smell name without extension
        smell_name = os.path.splitext(csv_file)[0]
        label_names.append(smell_name)
        
        # Load the file
        file_path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(file_path)
        
        try:
            # First clean column names to validate structure
            df = df.copy()
            df.columns = df.columns.str.strip()
            
            # Validate structure
            ok, missing = validate_structure(df)
            if not ok:
                print(f"Warning: File {csv_file} missing columns: {missing}")
                continue
                
            # Remove rows with Cycle Loop 0 and Mode "Resting"
            if 'Cycle Loop' in df.columns and 'Mode' in df.columns and not df[(df["Cycle Loop"] == 0) & (df["Mode"].str.strip() == "Resting")].empty:
                df = df.loc[~((df["Cycle Loop"] == 0) & (df["Mode"].str.strip() == "Resting"))]
                print(f"File {csv_file}: Removed rows with Cycle Loop 0 and Mode 'Resting'")
            
            # Analyze and handle cycles
            if 'Cycle Loop' in df.columns:
                too_short, too_long = analyze_cycles(df)
                
                # Drop short cycles
                if too_short:
                    df = drop_cycles(df, list(too_short.keys()))
                    print(f"File {csv_file}: Dropped {len(too_short)} short cycles.")
                
                # Trim long cycles
                if too_long:
                    df = trim_cycles(df)
                    print(f"File {csv_file}: Trimmed {len(too_long)} long cycles.")
            else:
                print(f"Warning: File {csv_file} doesn't have 'Cycle Loop' column")
                continue
            
            # Apply final preprocessing steps from pipeline_streamlit
            df = clean_columns(df)
            df = scale_features(df)  # This applies MinMaxScaler per cycle
            
            # Extract sensor columns (G1-G15)
            sensor_cols = [f'G{i}' for i in range(1, 16)]
            
            # Create segments by cycle
            cycles = df['Cycle Loop'].unique()
            for cycle in cycles:
                cycle_data = df[df['Cycle Loop'] == cycle][sensor_cols].values
                
                # All cycles should now be exactly cycle_length rows
                if len(cycle_data) == cycle_length:
                    all_features.append(cycle_data)
                    all_labels.append(label_idx)
        
        except Exception as e:
            print(f"Error processing file {csv_file}: {str(e)}")
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    # Note: We don't apply StandardScaler here since we've already applied MinMaxScaler
    # per cycle in the scale_features() step above
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create PyTorch datasets with appropriate model_type
    train_dataset = ENoseDataset(X_train, y_train, model_type=model_type)
    test_dataset = ENoseDataset(X_test, y_test, model_type=model_type)
    
    # Create label mapping
    label_map = {idx: name for idx, name in enumerate(label_names)}
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'label_map': label_map,
        'num_classes': len(label_names),
        'scaler': None  # We don't need to return a scaler since scaling is done per cycle
    }


def create_dataloaders(train_dataset, test_dataset, batch_size=32, num_workers=4):
    """
    Create DataLoader objects for training and testing.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Testing dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        
    Returns:
        Dictionary with train and test DataLoader objects
    """
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return {
        'train_loader': train_loader,
        'test_loader': test_loader
    }


def get_single_cycle_data(df, cycle, sensor_cols=None):
    """
    Extract data for a single cycle from a DataFrame.
    
    Args:
        df: DataFrame with sensor data
        cycle: Cycle number to extract
        sensor_cols: List of sensor column names
        
    Returns:
        Numpy array with sensor data for the specified cycle
    """
    if sensor_cols is None:
        sensor_cols = [f'G{i}' for i in range(1, 16)]
        
    # Extract cycle data
    cycle_data = df[df['Cycle Loop'] == cycle][sensor_cols].values
    
    return cycle_data


def preprocess_single_cycle(cycle_data, cycle_length=60, model_type='cnn'):
    """
    Preprocess a single cycle of data for model inference.
    Uses the same preprocessing approach as in pipeline_streamlit.py.
    
    Args:
        cycle_data: Numpy array with sensor readings for one cycle
        cycle_length: Expected cycle length (will truncate or pad)
        model_type: Type of model ('cnn' or 'lstm') to determine tensor format
        
    Returns:
        Preprocessed data ready for model input
    """
    # Ensure cycle data has the right length
    if len(cycle_data) > cycle_length:
        # Truncate to expected length
        cycle_data = cycle_data[:cycle_length]
    elif len(cycle_data) < cycle_length:
        # Pad with zeros to expected length
        padding = np.zeros((cycle_length - len(cycle_data), cycle_data.shape[1]))
        cycle_data = np.vstack([cycle_data, padding])
    
    # Apply MinMaxScaler like in the scale_features function
    scaler = MinMaxScaler()
    cycle_data = scaler.fit_transform(cycle_data)
    
    # Add batch dimension
    cycle_data = np.expand_dims(cycle_data, axis=0)
    
    # For CNN, transpose the dimensions
    if model_type == 'cnn':
        # Transpose from [batch, seq_length, features] to [batch, features, seq_length] for CNN
        cycle_data = np.transpose(cycle_data, (0, 2, 1))
    # For LSTM, keep as [batch, seq_length, features]
    
    return cycle_data
