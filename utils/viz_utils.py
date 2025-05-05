import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
import os

def plot_multiple_graphs(df, graphs, cycle, ncols=3, color_palette='viridis'):
    """
    Plot time-domain signals for a single cycle loop with enhanced visualization.
    
    Parameters:
    - df: Processed pandas DataFrame (must include a 'Cycle Loop' column)
    - graphs: List of column names to plot (e.g., ['G1', 'G2', …, 'G15'])
    - cycle: The cycle loop ID to visualize
    - ncols: Number of columns in the subplot grid (default is 3)
    - color_palette: Matplotlib colormap name for consistent color scheme
    """
    # Filter the DataFrame to only the selected cycle
    df_cycle = df[df["Cycle Loop"] == cycle]
    if df_cycle.empty:
        st.warning(f"No data available for cycle {cycle}")
        return

    # Create a modern style for plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Get a color palette for consistency
    colors = plt.cm.get_cmap(color_palette, len(graphs))
    
    # Determine grid size
    n = len(graphs)
    nrows = (n + ncols - 1) // ncols
    
    # Create figure with better resolution
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4 * nrows),
        dpi=120,
        squeeze=False
    )
    axes = axes.flatten()
    
    # Add a title for the entire figure
    fig.suptitle(f"Signal Analysis for Cycle Loop {cycle}", 
                 fontsize=16, fontweight="bold", y=0.98)
    
    # Generate x-axis values (time)
    time_vals = np.arange(len(df_cycle))
    
    # Plot each selected sensor signal with enhanced styling
    for i, (ax, col) in enumerate(zip(axes, graphs)):
        # Plot with gradient alpha for depth effect
        line = ax.plot(time_vals, df_cycle[col].values, 
                       label=col, 
                       color=colors(i/len(graphs)),
                       linewidth=2.0,
                       alpha=0.9)[0]
        
        # Add slight shading under the curve
        ax.fill_between(time_vals, df_cycle[col].values, 
                         alpha=0.2, color=line.get_color())
        
        # Improve axis labels and titles
        ax.set_xlabel("Time Steps", fontsize=11)
        ax.set_ylabel("Amplitude", fontsize=11)
        ax.set_title(f"{col} Signal", fontsize=13, fontweight="bold")
        
        # Add legend with better positioning
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
        
        # Improve grid appearance
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Use integer ticks on x-axis where appropriate
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add light background shading for better readability
        ax.set_facecolor('#f8f9fa')
        
        # Add subtle box around the plot
        for spine in ax.spines.values():
            spine.set_edgecolor('#dddddd')
            
        # Add data statistics as text
        stats_text = (f"Min: {df_cycle[col].min():.2f}\n"
                     f"Max: {df_cycle[col].max():.2f}\n"
                     f"Mean: {df_cycle[col].mean():.2f}")
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
                fontsize=8, va='bottom', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='white', alpha=0.7))

    # Hide any unused subplots
    for ax in axes[n:]:
        ax.axis("off")

    # Improve overall layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
    
    # Add subtle footer with timestamp
    fig.text(0.5, 0.01, f"Generated with Streamlit Analysis Tool", 
             ha='center', fontsize=9, fontstyle='italic', color='#666666')
    
    # Display the figure in Streamlit
    st.pyplot(fig)

def plot_sensor_statistics(cycle_data, sensor_cols):
    """
    Create and display a summary DataFrame of sensor statistics.
    
    Parameters:
    - cycle_data: DataFrame filtered to a specific cycle
    - sensor_cols: List of sensor column names to analyze
    
    Returns:
    - summary: DataFrame containing the statistics
    """
    # Calculate statistics for each sensor
    summary = pd.DataFrame({
        'Sensor': sensor_cols,
        'Min': [cycle_data[col].min() for col in sensor_cols],
        'Max': [cycle_data[col].max() for col in sensor_cols],
        'Mean': [cycle_data[col].mean() for col in sensor_cols],
        'Std Dev': [cycle_data[col].std() for col in sensor_cols]
    })
    
    # Display the summary dataframe
    st.dataframe(summary)
    
    return summary

def plot_prediction_results(top_predictions):
    """
    Create a horizontal bar chart showing top prediction results.
    
    Parameters:
    - top_predictions: List of dictionaries with 'label' and 'probability' keys
    
    Returns:
    - fig: The matplotlib figure object
    """
    # Create visualization with matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [p["label"] for p in top_predictions]
    probs = [p["probability"] for p in top_predictions]
    colors = ["#f37c45", "#ffbb57", "#ffd57b"]
    
    bars = ax.barh(labels, probs, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Top 3 Smell Predictions", fontsize=14)
    
    # Add probability values
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
                f"{width:.1f}%", va='center')
    
    plt.tight_layout()
    
    # Display the figure in Streamlit
    st.pyplot(fig)
    
    return fig

# Functions from train_model.py
def plot_metrics(history, save_path=None, model_name=None, version=None):
    """
    Plot training and validation metrics.
    
    Parameters:
    - history: Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    - save_path: Path to save the plot (optional)
    - model_name: Model name for saving to standardized folder (optional)
    - version: Model version for saving to standardized folder (optional)
    
    Returns:
    - fig: The matplotlib figure object
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot using the standardized function if model info is provided
    if model_name and version:
        fig = plt.gcf()  # Get current figure
        fig_path = save_plot_to_folder(fig, model_name, version, 'metrics')
    else:
        fig = plt.gcf()  # Get current figure
        if save_path:
            plt.savefig(save_path)
            print(f"Metrics plot saved to: {save_path}")
    
    return fig

def plot_confusion_matrix(cm, class_names, save_path=None, model_name=None, version=None):
    """
    Plot confusion matrix.
    
    Parameters:
    - cm: Confusion matrix array
    - class_names: List of class names
    - save_path: Path to save the plot (optional)
    - model_name: Model name for saving to standardized folder (optional)
    - version: Model version for saving to standardized folder (optional)
    
    Returns:
    - fig: The matplotlib figure object
    """
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add class labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
    plt.yticks(tick_marks, class_names, fontsize=6)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", 
                     fontsize=6)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the plot using the standardized function if model info is provided
    if model_name and version:
        fig = plt.gcf()  # Get current figure
        fig_path = save_plot_to_folder(fig, model_name, version, 'confusion_matrix')
    else:
        fig = plt.gcf()  # Get current figure
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix plot saved to: {save_path}")
    
    return fig

def save_plot_to_folder(fig, model_name, version, plot_type, base_dir='plots'):
    """
    Save a plot to a properly organized folder with standardized naming.
    
    Parameters:
    - fig: The matplotlib figure to save
    - model_name: Model name (e.g., 'cnn', 'lstm', 'transformer')
    - version: Model version (e.g., 'v1', 'v2')
    - plot_type: Type of plot (e.g., 'metrics', 'confusion_matrix')
    - base_dir: Base directory for all plots
    
    Returns:
    - filepath: Path to the saved plot
    """
    # Create the model-specific directory if it doesn't exist
    plot_dir = os.path.join(base_dir, model_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create the filename with requested format
    filename = f"{model_name}_{version}_{plot_type}.png"
    filepath = os.path.join(plot_dir, filename)
    
    # Save the figure
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filepath}")
    
    return filepath