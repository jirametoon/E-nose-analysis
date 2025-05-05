# E-NOSE: Electronic Nose Analytics

A machine learning application for electronic nose (E-NOSE) smell classification using PyTorch and Streamlit.

![E-Nose Logo](assets/logo_nose.png)

## Project Overview

This application uses deep learning models to analyze and classify odors captured by an electronic nose device equipped with gas sensors. The system provides visualization, data analysis, and classification of different smells using CNN, LSTM, and Transformer models.

## Features

- Interactive web interface using Streamlit
- Support for multiple deep learning model architectures (CNN, LSTM, Transformer)
- Real-time smell classification
- Data visualization and analysis tools
- Model training and evaluation capabilities

## Getting Started

### Prerequisites

- Python 3.8+ 
- CUDA-compatible GPU (optional but recommended for training)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jirametoon/E-nose.git
   cd e-nose-analytics
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

The repository already includes pre-trained models in the `models/` directory.

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

### Training Models

If you want to retrain the models or train with your own data:

```bash
python train_model.py --model cnn --data_dir data/15cycle_datasets --epochs 200
```

See `python train_model.py --help` for all options.

## Project Structure

```
app.py                  # Main Streamlit application
train_model.py          # Model training script
requirements.txt        # Python dependencies
models/                 # Model architecture definitions and saved models
  ├── cnn.py
  ├── lstm.py
  ├── transformer.py
  ├── cnn/              # Saved CNN models
  ├── lstm/             # Saved LSTM models
  └── transformer/      # Saved Transformer models
data/                   # Dataset files
  └── 15cycle_datasets/ # Training data
utils/                  # Utility functions
  ├── data_utils.py     # Data loading and preprocessing
  ├── inference.py      # Model inference
  ├── page_utils.py     # UI pages
  └── viz_utils.py      # Visualization utilities
assets/                 # Images and static files
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.