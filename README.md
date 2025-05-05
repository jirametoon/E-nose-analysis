# E-NOSE: Electronic Nose Analytics

A machine learning application for electronic nose (E-NOSE) smell classification using PyTorch and Streamlit.

![E-Nose Logo](assets/logo_nose.png)

## Project Overview

This application uses deep learning models to analyze and classify odors captured by an electronic nose device equipped with gas sensors. The system provides visualization, data analysis, and classification of different smells using CNN, LSTM, and Transformer models.

> **Note:** This project was created for fun with my friends and is not intended for research purposes. Please feel free to use it and train models however you want!

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
   git clone https://github.com/jirametoon/E-nose-analysis.git
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

### Evaluating Models

To evaluate a trained model on test data:

```bash
python evaluate_model.py --model cnn --model_path models/cnn/cnn_v1_best.pt --data_dir data/15cycle_datasets
```

This will generate evaluation metrics including accuracy, precision, recall, F1-score, and a confusion matrix in the `plots` directory.

Additional options:
```bash
python evaluate_model.py --help
```

### Comparing Models

To compare the performance of different model architectures:

```bash
python compare_models.py --models cnn lstm transformer --data_dir data/15cycle_datasets
```

This will generate comparative visualizations and performance metrics for the specified models. You can also compare specific model versions:

```bash
python compare_models.py --models cnn lstm --model_paths models/cnn/cnn_v1_best.pt models/lstm/lstm_v1_best.pt
```

See all options:
```bash
python compare_models.py --help
```

### Data Format

The input data for the models should be CSV files with the following structure:

- **Headers**: Date, Time, Temp, Flow Rate, G1-G15 (sensor readings), Cycle Loop, Mode
- **Modes**: The data cycles through three phases: Resting → Sampling → Purging → Resting
- **Sample File**: A sample CSV file is included in `data/samples/sample.csv` to demonstrate the expected format

Example format:
```
Date,Time,Temp,Flow Rate,G1,G2,...,G15,Cycle Loop,Mode
1/5/2023,16:44:30,55.119,299.865,0,-1.6,...,-6,0,Resting
1/5/2023,16:44:46,55.110,299.865,-2,-7.6,...,-24.8,1,Sampling
1/5/2023,16:45:16,55.021,299.865,87.4,287.8,...,1876,1,Purging
```

Each odor class should be represented by a separate CSV file named with the format `[OdorName]_final.csv`.

## Project Structure

```
app.py                  # Main Streamlit application
train_model.py          # Model training script
evaluate_model.py       # Model evaluation script
compare_models.py       # Script for comparing different models
requirements.txt        # Python dependencies
models/                 # Model architecture definitions and saved models
  ├── cnn.py
  ├── lstm.py
  ├── transformer.py
  ├── cnn/              # Saved CNN models
  ├── lstm/             # Saved LSTM models
  └── transformer/      # Saved Transformer models
data/                   # Dataset files
  ├── datasets/         # Training data
  └── samples/          # Sample data format for reference
utils/                  # Utility functions
  ├── data_utils.py     # Data loading and preprocessing
  ├── inference.py      # Model inference
  ├── page_utils.py     # UI pages
  └── viz_utils.py      # Visualization utilities
assets/                 # Images and static files
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.