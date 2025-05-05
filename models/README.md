# E-NOSE Models

This directory contains the deep learning model architectures and pre-trained models for the E-NOSE smell classification system.

## Model Architectures

The E-NOSE system supports three different deep learning architectures:

### 1. CNN (Convolutional Neural Network)

`cnn.py` implements a 1D-CNN architecture specifically designed for processing sensor array time series data.

- **Architecture**: Multiple 1D convolutional layers with batch normalization, max-pooling, and dropout
- **Input**: Tensor of shape [batch_size, 15, 60] representing 15 gas sensors (G1-G15) over 60 time steps
- **Key Strengths**: Effective at capturing local patterns in sensor response curves

### 2. LSTM (Long Short-Term Memory)

`lstm.py` implements an LSTM architecture for sequential sensor data processing.

- **Architecture**: Bidirectional LSTM with multiple layers, followed by fully connected layers
- **Input**: Tensor of shape [batch_size, 60, 15] representing 60 time steps of 15 gas sensors
- **Key Strengths**: Excellent at capturing temporal dependencies and long-term patterns

### 3. Transformer

`transformer.py` implements a Transformer-based architecture using self-attention mechanisms.

- **Architecture**: Multi-head self-attention with position encoding and feed-forward networks
- **Input**: Tensor of shape [batch_size, 60, 15] representing 60 time steps of 15 gas sensors
- **Key Strengths**: Captures global dependencies regardless of distance in the sequence

## Pre-trained Models

The `models/cnn/`, `models/lstm/`, and `models/transformer/` directories contain pre-trained model checkpoints:

- Format: `{model_type}_v{version}_{epoch}.pt` (e.g., `cnn_v1_100.pt` = CNN model v1 after 100 epochs)
- Best models: `{model_type}_v{version}_best.pt` (model with the highest validation accuracy)

## Usage Examples

### Training Models

```bash
# Train a CNN model for 200 epochs
python train_model.py --model cnn --data_dir data/15cycle_datasets --epochs 200 --model_name cnn_v1

# Train an LSTM model with custom batch size and learning rate
python train_model.py --model lstm --data_dir data/15cycle_datasets --epochs 150 --batch_size 64 --lr 0.0005 --model_name lstm_v1

# Train a Transformer model using GPU with specific cycle length
python train_model.py --model transformer --data_dir data/15cycle_datasets --epochs 250 --device cuda --cycle_length 60 --model_name transformer_v1
```

### Evaluating Models

```bash
# Evaluate the best CNN model
python evaluate_model.py --model_path models/cnn/cnn_v1_best.pt --data_dir data/15cycle_datasets

# Compare performance between model types
python compare_models.py --models models/cnn/cnn_v1_best.pt models/lstm/lstm_v1_best.pt models/transformer/transformer_v1_best.pt
```

### Inference with Pre-trained Models

```bash
# Perform inference on a single sample
python inference.py --model_path models/cnn/cnn_v1_best.pt --input_file unseen_example/Rasberry1.csv
```

## Model Performance Comparison

| Model      | Accuracy  |
|------------|-----------|
| CNN        | 63.04%    |
| LSTM       | 26.81%    |
| Transformer| 39.86%    |

*Note: Performance metrics are based on our standard test set using a CUDA-capable GPU.*

## Model Selection Guide

- **CNN**: Best for real-time applications where inference speed is critical
- **LSTM**: Good balance between accuracy and computational requirements
- **Transformer**: Highest accuracy but more computationally intensive

## Extending the Models

To add a new model architecture:

1. Create a new file `models/your_model.py` implementing a PyTorch `nn.Module` class
2. Add the model to `MODEL_CLASSES` in `train_model.py`
3. Implement the required interface methods (`forward`, `predict`, `preprocess_data`)

See existing model implementations for reference.

## Citation

If you use these models in your research, please cite:

```
@article{enose2025,
  title={Deep Learning Architectures for Electronic Nose Data Classification},
  author={E-NOSE Team},
  journal={Journal of Artificial Olfaction},
  year={2025}
}
```