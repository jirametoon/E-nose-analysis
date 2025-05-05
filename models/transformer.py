import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for the Transformer model.
    
    Adds information about the relative or absolute position of tokens in a sequence.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
            Tensor with added positional encodings
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerNet(nn.Module):
    """Transformer model for time series classification.
    
    Especially suited for complex pattern recognition in E-NOSE data,
    capturing both local and global dependencies between sensor readings.
    """
    
    def __init__(self, input_dim=15, d_model=64, nhead=4, 
                 num_layers=2, dim_feedforward=128, dropout=0.1, 
                 num_classes=45):
        """
        Initialize the Transformer model.
        
        Args:
            input_dim (int): Number of input features (default: 15 for G1-G15 sensors)
            d_model (int): Dimension of model embeddings
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer encoder layers
            dim_feedforward (int): Dimension of feedforward network in transformer
            dropout (float): Dropout probability for regularization
            num_classes (int): Number of smell classes to predict
        """
        super(TransformerNet, self).__init__()
        
        # Project input features to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Add positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, 128)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
                representing sensor readings across time
                
        Returns:
            Tensor of shape [batch_size, num_classes] with class probabilities
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling over sequence length
        x = torch.mean(x, dim=1)
        
        # Output layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, x):
        """
        Get class predictions from model.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class indices
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs, 1)
        return predicted
    
    @staticmethod
    def preprocess_data(data, device):
        """
        Preprocess input data for the model.
        
        Args:
            data: Numpy array of shape [batch_size, seq_length, input_dim]
            device: PyTorch device (cuda or cpu)
            
        Returns:
            PyTorch tensor ready for the model
        """
        # Convert to tensor if not already
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        # Ensure correct shape [batch_size, seq_length, input_dim]
        if data.dim() == 2:  # single sample: [seq_length, input_dim]
            data = data.unsqueeze(0)
            
        return data.to(device)