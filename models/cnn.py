import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    """1D Convolutional Neural Network for classifying sensor array data.
    
    This model is designed specifically for electronic nose (E-NOSE) data
    with 15 gas sensor inputs arranged as a time series.
    """
    
    def __init__(self, input_channels=15, seq_length=60, num_classes=45):
        """
        Initialize the 1D CNN model.
        
        Args:
            input_channels (int): Number of input channels (default: 15 for G1-G15 sensors)
            seq_length (int): Length of each input sequence (default: 60 rows per cycle)
            num_classes (int): Number of smell classes to predict
        """
        super(CNN1D, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        # After 3 pooling layers with stride 2: seq_length // 2^3
        conv_output_size = 256 * (seq_length // 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, input_channels, seq_length]
                representing sensor readings across time
                
        Returns:
            Tensor of shape [batch_size, num_classes] with class probabilities
        """
        # Convolutional blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
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
            data: Numpy array of shape [batch_size, seq_length, input_channels]
            device: PyTorch device (cuda or cpu)
            
        Returns:
            PyTorch tensor ready for the model
        """
        # Convert to tensor if not already
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        # Reshape data to [batch_size, input_channels, seq_length]
        if data.dim() == 3:  # [batch_size, seq_length, input_channels]
            data = data.transpose(1, 2)
        elif data.dim() == 2:  # single sample: [seq_length, input_channels]
            data = data.transpose(0, 1).unsqueeze(0)
            
        return data.to(device)