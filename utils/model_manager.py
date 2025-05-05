import os
import streamlit as st
from pathlib import Path
import time
import gdown
import logging
import glob

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_manager")

# Define model storage locations
MODEL_DIRS = {
    'cnn': 'models/cnn',
    'lstm': 'models/lstm',
    'transformer': 'models/transformer',
    'my_model': 'models/my_model'  # Add custom model folder
}

# Model type mapping for display and folder names
MODEL_TYPE_MAPPING = {
    'cnn': 'CNN1D',
    'lstm': 'LSTMNet',
    'transformer': 'TransformerNet',
    'my_model': 'MyModel'  # Add custom model type mapping
}

# Reverse mapping
FOLDER_TO_MODEL_TYPE = {v: k for k, v in MODEL_TYPE_MAPPING.items()}

# Define Google Drive folder details
GOOGLE_DRIVE_FOLDER_ID = "1VivggUZlxmUub75B7CHZ0sP_JfL_pU3k"
GOOGLE_DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{GOOGLE_DRIVE_FOLDER_ID}"

# Define direct download links for model files - these should work properly when deployed
MODEL_DIRECT_LINKS = {
    'cnn': {
        'cnn_v1_best.pt': "https://drive.google.com/uc?export=download&id=15GsROi3ceyQ9jZN-X8XNDpH_yvDF1cws&confirm=t"
    },
    'lstm': {
        'lstm_v1_best.pt': "https://drive.google.com/uc?export=download&id=1tLySLl3aQCHjs5xlf0KLzlIFOShaGw9n&confirm=t"
    }, 
    'transformer': {
        'transformer_v1_best.pt': "https://drive.google.com/uc?export=download&id=1_to5z5FoIjy9-5jFFJ5VbNaf0xGtyWjb&confirm=t"
    }
}

# Store Google Drive file list to avoid repeated API calls
GDRIVE_MODEL_CACHE = {
    'last_updated': None,
    'models': {
        'cnn': [],
        'lstm': [],
        'transformer': [],
        'my_model': []  # Add my_model to cache
    }
}

# Cache duration in seconds (10 minutes)
CACHE_DURATION = 600

def check_model_exists(model_type):
    """Check if model files exist locally"""
    model_dir = Path(MODEL_DIRS[model_type])
    # Create directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    # Check for any .pt files in the directory
    model_files = list(model_dir.glob("*.pt"))
    return len(model_files) > 0

def get_google_drive_models():
    """Get a list of available models from Google Drive"""
    # Check if cache is still valid
    if (GDRIVE_MODEL_CACHE['last_updated'] and 
        time.time() - GDRIVE_MODEL_CACHE['last_updated'] < CACHE_DURATION):
        return GDRIVE_MODEL_CACHE['models']
    
    # Clear the current cache
    for model_type in GDRIVE_MODEL_CACHE['models']:
        GDRIVE_MODEL_CACHE['models'][model_type] = []
    
    # First get local models - these are confirmed to exist
    for model_type, model_dir in MODEL_DIRS.items():
        local_dir = Path(model_dir)
        if local_dir.exists():
            local_models = list(local_dir.glob("*.pt"))
            for model in local_models:
                if model.name not in GDRIVE_MODEL_CACHE['models'][model_type]:
                    GDRIVE_MODEL_CACHE['models'][model_type].append(model.name)
    
    # Always include expected Google Drive models, even if we already have local models
    # These are the models you mentioned having (one per type)
    GDRIVE_MODEL_CACHE['models']['cnn'].append('cnn_v1_best.pt')
    GDRIVE_MODEL_CACHE['models']['lstm'].append('lstm_v1_best.pt')
    GDRIVE_MODEL_CACHE['models']['transformer'].append('transformer_v1_best.pt')
    
    # Update cache timestamp
    GDRIVE_MODEL_CACHE['last_updated'] = time.time()
    return GDRIVE_MODEL_CACHE['models']

def download_model_file(model_type, model_filename):
    """Download a specific model file from Google Drive"""
    import requests
    from tqdm.auto import tqdm
    
    # Create directory if it doesn't exist
    model_dir = Path(MODEL_DIRS[model_type])
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output path for the model file
    model_path = model_dir / model_filename
    
    # Check if already downloaded
    if model_path.exists():
        logger.info(f"Model {model_filename} already exists locally")
        return True
    
    try:
        # Create a progress bar display in Streamlit
        with st.spinner(f"Downloading {model_filename} from Google Drive..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Try to get the model file using a direct download link
            if model_type in MODEL_DIRECT_LINKS and model_filename in MODEL_DIRECT_LINKS[model_type]:
                direct_url = MODEL_DIRECT_LINKS[model_type][model_filename]
                status_text.text(f"Initiating download for {model_filename}...")
                
                # Stream the response to file with progress tracking
                with requests.get(direct_url, stream=True) as response:
                    # Raise an exception for HTTP errors
                    response.raise_for_status()
                    
                    # Get content length if available for progress tracking
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    # Create model file and stream content
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:  # filter out keep-alive chunks
                                f.write(chunk)
                                downloaded += len(chunk)
                                
                                if total_size > 0:
                                    # Update progress as percentage
                                    progress = min(downloaded / total_size, 1.0)
                                    progress_bar.progress(progress)
                                    status_text.text(f"Downloading: {progress:.1%} complete")
                
                progress_bar.progress(1.0)
                status_text.text(f"Download complete!")
                st.success(f"Successfully downloaded {model_filename} from Google Drive!")
                return True
                
            else:
                # Try using standard gdown as fallback
                status_text.text("Direct download link not available. Trying gdown...")
                try:
                    success = gdown.download(
                        url=f"{GOOGLE_DRIVE_FOLDER_URL}/{model_type}/{model_filename}",
                        output=str(model_path),
                        quiet=False,
                        fuzzy=True,
                        resume=False
                    )
                    if success:
                        progress_bar.progress(1.0)
                        status_text.text("Download complete!")
                        st.success(f"Successfully downloaded {model_filename}!")
                        return True
                except Exception as gdown_error:
                    logger.error(f"gdown download failed: {gdown_error}")
                    status_text.text("gdown download failed.")
                
                # If we got here, all automatic methods failed
                progress_bar.empty()
                status_text.empty()
                show_manual_download_instructions(model_type, model_filename, model_path)
                return False
                
    except Exception as e:
        logger.error(f"Error downloading model {model_filename}: {e}")
        st.error(f"Could not download the model file. Error: {str(e)}")
        show_manual_download_instructions(model_type, model_filename, model_path)
        return False

def show_manual_download_instructions(model_type, model_filename, model_path):
    """Show instructions for manually downloading a model file"""
    # Create folder mappings for clearer instructions
    folder_display_names = {
        'cnn': 'CNN1D',
        'lstm': 'LSTM', 
        'transformer': 'Transformer'
    }
    
    st.warning("Automatic model downloads failed. Please download the model manually.")
    
    st.info(f"""
    ### Manual Download Instructions:
    
    1. Open the [E-nose Models Google Drive folder](https://drive.google.com/drive/folders/{GOOGLE_DRIVE_FOLDER_ID})
    2. Navigate to the **{folder_display_names.get(model_type, model_type)}** folder
    3. Download **{model_filename}**
    4. Save it to: `{model_path.absolute()}`
    5. Restart the application after downloading
    """)
    
    # Create a download button as well
    download_link = f"https://drive.google.com/drive/folders/{GOOGLE_DRIVE_FOLDER_ID}/{folder_display_names.get(model_type, model_type)}"
    st.markdown(f"[⬇️ Open Google Drive Folder]({download_link})")
    
    return False

def get_available_models(include_gdrive=True):
    """Get all available model types and versions, both local and from Google Drive"""
    models = {
        "CNN1D": [],
        "LSTMNet": [],
        "TransformerNet": []
    }
    
    # First, check local models
    for model_type, model_dir in MODEL_DIRS.items():
        display_type = MODEL_TYPE_MAPPING[model_type]
        local_dir = Path(model_dir)
        if local_dir.exists():
            # Find all local .pt files
            model_files = [f.name for f in local_dir.glob("*.pt")]
            for file in model_files:
                # Always add local models with the local_ prefix
                models[display_type].append(f"local_{file}")
    
    # Then, if enabled, check Google Drive models
    if include_gdrive:
        try:
            gdrive_models = get_google_drive_models()
            
            # The key models we always want to show as available from Google Drive (one per type)
            required_gdrive_models = {
                'cnn': 'cnn_v1_best.pt',
                'lstm': 'lstm_v1_best.pt', 
                'transformer': 'transformer_v1_best.pt'
            }
            
            for model_type, file_list in gdrive_models.items():
                display_type = MODEL_TYPE_MAPPING[model_type]
                
                # Always add the "required" Google Drive model first (if specified for this type)
                if model_type in required_gdrive_models:
                    required_file = required_gdrive_models[model_type]
                    models[display_type].append(f"gdrive_{required_file}")
                
                # Then add any other models that aren't duplicates
                for file in file_list:
                    local_version = f"local_{file}"
                    gdrive_version = f"gdrive_{file}"
                    # Skip if it's already added as a required model or as a local version
                    if gdrive_version not in models[display_type] and local_version not in models[display_type]:
                        models[display_type].append(gdrive_version)
        except Exception as e:
            logger.error(f"Error getting Google Drive models: {e}")
            st.warning("Could not check Google Drive for models. Using local models only.")
    
    return models

def ensure_model_available(model_type, model_version=None):
    """Ensure the specified model is available, downloading from GDrive if necessary"""
    # Normalize model_type to folder name if needed
    folder_name = model_type.lower()
    if model_type in MODEL_TYPE_MAPPING.values():
        folder_name = next(k for k, v in MODEL_TYPE_MAPPING.items() if v == model_type)
    
    # If no specific version requested, use any available model
    if model_version is None:
        if check_model_exists(folder_name):
            return True
            
        # Try to download "best" model
        model_files = get_google_drive_models()[folder_name]
        for model_file in model_files:
            if 'best' in model_file:
                return download_model_file(folder_name, model_file)
        
        # If no "best" model, download the first available
        if model_files:
            return download_model_file(folder_name, model_files[0])
        
        st.error(f"No {model_type} models found on Google Drive")
        return False
        
    # Handling specific model version
    else:
        # Check if it's a Google Drive model (prefixed with gdrive_)
        if model_version.startswith("gdrive_"):
            actual_filename = model_version[7:]  # Remove the "gdrive_" prefix
            return download_model_file(folder_name, actual_filename)
            
        # It's a local model, check if it exists
        model_dir = Path(MODEL_DIRS[folder_name])
        if model_version.startswith("local_"):
            actual_filename = model_version[6:]  # Remove the "local_" prefix
            return (model_dir / actual_filename).exists()
        
        # No prefix, assume it's a local file
        return (model_dir / model_version).exists()

def get_model_path(model_type, model_version=None):
    """Get the path to the model file of the specified type and version"""
    # Normalize model_type to folder name if needed
    folder_name = model_type.lower()
    if model_type in MODEL_TYPE_MAPPING.values():
        folder_name = next(k for k, v in MODEL_TYPE_MAPPING.items() if v == model_type)
    
    # If model version is specified with a prefix
    if model_version:
        if model_version.startswith("gdrive_"):
            # Ensure the Google Drive model is downloaded
            actual_filename = model_version[7:]  # Remove "gdrive_" prefix
            if ensure_model_available(folder_name, model_version):
                return str(Path(MODEL_DIRS[folder_name]) / actual_filename)
            return None
            
        elif model_version.startswith("local_"):
            # Local model
            actual_filename = model_version[6:]  # Remove "local_" prefix
            model_path = Path(MODEL_DIRS[folder_name]) / actual_filename
            if model_path.exists():
                return str(model_path)
            return None
        
        # No prefix, assume it's a local file
        model_path = Path(MODEL_DIRS[folder_name]) / model_version
        if model_path.exists():
            return str(model_path)
            
    # No specific version, find best available
    model_dir = Path(MODEL_DIRS[folder_name])
    
    # First, check for best model
    best_model = list(model_dir.glob("*best*.pt"))
    if best_model:
        return str(best_model[0])
        
    # Then check for any model
    any_model = list(model_dir.glob("*.pt"))
    if any_model:
        return str(any_model[0])
    
    return None

def use_direct_gdrive_download():
    """Creates local models from Google Drive directly to bypass gdown issues"""
    st.markdown("""
    ## Manual Model Setup
    
    The automatic model detection from Google Drive isn't working properly. Please follow these steps:
    
    1. Open the [E-nose Models Google Drive folder](https://drive.google.com/drive/folders/1VivggUZlxmUub75B7CHZ0sP_JfL_pU3k)
    2. For each model type (CNN, LSTM, Transformer):
        - Go into the folder
        - Download the model files (*.pt)
        - Place them in the corresponding local folder
    """)
    
    # Show the local paths where models should be placed
    for model_type, folder in MODEL_DIRS.items():
        full_path = os.path.abspath(folder)
        # Create the directory if it doesn't exist
        os.makedirs(full_path, exist_ok=True)
        st.info(f"Place {model_type.upper()} models in: `{full_path}`")
    
    # Let user know model directories have been created
    st.success("Model directories have been created. After placing model files, restart the app.")
    
    # Return the model directories for reference
    return {model_type: os.path.abspath(folder) for model_type, folder in MODEL_DIRS.items()}