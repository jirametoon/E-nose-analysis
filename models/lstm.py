import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    """Long Short-Term Memory (LSTM) network for time series classification.
    
    Designed specifically for electronic nose (E-NOSE) data with multiple sensor inputs.
    LSTMs are particularly good at capturing temporal dependencies in sensor response patterns.
    """
    
    def __init__(self, input_size=15, hidden_size=128, num_layers=2, 
                 dropout=0.3, num_classes=45):
        """
        Initialize the LSTM model.
        
        Args:
            input_size (int): Number of input features (default: 15 for G1-G15 sensors)
            hidden_size (int): Number of features in the hidden state
            num_layers (int): Number of recurrent layers
            dropout (float): Dropout probability for regularization
            num_classes (int): Number of smell classes to predict
        """
        super(LSTMNet, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_size]
                representing sensor readings across time
                
        Returns:
            Tensor of shape [batch_size, num_classes] with class probabilities
        """
        # Initialize hidden and cell states
        batch_size = x.size(0)
        
        # LSTM forward pass
        # Output shape: [batch_size, seq_length, hidden_size*2] (bidirectional)
        lstm_out, _ = self.lstm(x)
        
        # We use the output from the last time step for classification
        # Shape: [batch_size, hidden_size*2]
        final_hidden_state = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.relu(self.fc1(final_hidden_state))
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
            data: Numpy array of shape [batch_size, seq_length, input_size]
            device: PyTorch device (cuda or cpu)
            
        Returns:
            PyTorch tensor ready for the model
        """
        # Convert to tensor if not already
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        # Ensure correct shape [batch_size, seq_length, input_size]
        if data.dim() == 2:  # single sample: [seq_length, input_size]
            data = data.unsqueeze(0)
            
        return data.to(device)