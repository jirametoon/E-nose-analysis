"""
Result page implementation for the E-nose Analytics application.
"""
import streamlit as st
import random
import os
from datetime import datetime

from .styles import get_result_styles
from ..viz_utils import (
    plot_multiple_graphs, 
    plot_sensor_statistics,
    plot_prediction_results
)

def get_available_models():
    """Get all available model types and versions"""
    models = {
        "CNN1D": [],
        "LSTMNet": [],
        "TransformerNet": []
    }
    
    # Map folder names to model types
    folder_mapping = {
        "cnn": "CNN1D",
        "lstm": "LSTMNet",
        "transformer": "TransformerNet"
    }
    
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    
    # Scan model directories
    for folder_name, model_type in folder_mapping.items():
        folder_path = os.path.join(base_path, folder_name)
        if os.path.exists(folder_path):
            model_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
            models[model_type] = sorted(model_files)
    
    return models

def render():
    """
    Renders the results page with model predictions and visualizations.
    """
    # Apply custom CSS
    st.markdown(get_result_styles(), unsafe_allow_html=True)
    
    # Main container
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    
    # Get URL parameters
    query_params = st.query_params
    
    # Case 1: Coming from Process page via Submit button
    if "analysis_ready" in st.session_state and st.session_state["analysis_ready"]:
        cycle_loop = st.session_state.get("selected_cycle", "N/A")
        
        # Animated header with pulse effect
        st.markdown(f"""
        <div class="header-wrapper">
            <h1 class="result-header animated-gradient">
                <span class="header-icon">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M22 11.08V12C21.9988 14.1564 21.3005 16.2547 20.0093 17.9818C18.7182 19.709 16.9033 20.9725 14.8354 21.5839C12.7674 22.1953 10.5573 22.1219 8.53447 21.3746C6.51168 20.6273 4.78465 19.2461 3.61096 17.4371C2.43727 15.628 1.87979 13.4881 2.02168 11.3363C2.16356 9.18455 2.99721 7.13631 4.39828 5.49706C5.79935 3.85781 7.69279 2.71537 9.79619 2.24013C11.8996 1.7649 14.1003 1.98232 16.07 2.85999" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M22 4L12 14.01L9 11.01" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </span>
                Analysis Results
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Display basic cycle information with modern cards
        st.markdown("""
        <div class="info-panel glass-card">
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-icon">🔄</div>
                    <div>
                        <div class="info-label">Selected Cycle</div>
                        <div class="info-value">""" + str(cycle_loop) + """</div>
                    </div>
                </div>
                <div class="info-item">
                    <div class="info-icon">📅</div>
                    <div>
                        <div class="info-label">Analysis Date</div>
                        <div class="info-value">""" + datetime.now().strftime("%b %d, %Y") + """</div>
                    </div>
                </div>
                <div class="info-item">
                    <div class="info-icon">⏱️</div>
                    <div>
                        <div class="info-label">Process Time</div>
                        <div class="info-value">""" + str(random.randint(2, 15)) + """ sec</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sensor readings for the selected cycle FIRST
        if "df_processed" in st.session_state:
            df = st.session_state["df_processed"]
            cycle_data = df[df["Cycle Loop"] == cycle_loop]
            sensor_cols = [f"G{i}" for i in range(1,16)]
            
            # Create a tab container with custom styling
            st.markdown('<div class="custom-tabs-container">', unsafe_allow_html=True)
            st.markdown('<h2 class="result-subheader with-icon"><span class="subheader-icon">📊</span> Sensor Readings Analysis</h2>', unsafe_allow_html=True)
            
            # Create tabs for different visualizations with better styling
            tab_labels = ["Sensor Signals", "Sensor Statistics"]
            tabs = st.tabs(tab_labels)
            
            with tabs[0]:
                st.markdown('<div class="tab-content">', unsafe_allow_html=True)
                # Plot all sensors (G1-G15) using the imported function
                plot_multiple_graphs(df, sensor_cols, cycle=cycle_loop)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tabs[1]:
                st.markdown('<div class="tab-content">', unsafe_allow_html=True)
                # Use imported functions for statistics
                plot_sensor_statistics(cycle_data, sensor_cols)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close custom-tabs-container
        
        # THEN show model selection after the data display
        st.markdown('<h2 class="result-subheader with-icon"><span class="subheader-icon">🧠</span> Model Selection</h2>', unsafe_allow_html=True)
        
        # Model selection
        available_models = get_available_models()
        
        # Create a horizontal layout with two equal columns
        col1, col2 = st.columns(2)
        
        with col1:
            MODEL_OPTIONS = {
                "CNN1D": "1D Convolutional Neural Network",
                "LSTMNet": "Long Short-Term Memory Network",
                "TransformerNet": "Transformer Neural Network"
            }
            
            # Custom HTML for prettier model selection
            st.markdown('<div class="select-container">', unsafe_allow_html=True)
            st.markdown('<label class="select-label">Model Architecture</label>', unsafe_allow_html=True)
            model_type = st.selectbox(
                "Model Architecture", 
                list(MODEL_OPTIONS.keys()), 
                format_func=lambda x: MODEL_OPTIONS[x],
                index=0,
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Use default model if none available
            if not available_models[model_type]:
                model_version = "Default"
                st.warning(f"No trained models found for {model_type}. Using default model.")
            else:
                # Clean model names for display
                model_versions = available_models[model_type]
                
                # Extract version numbers for slider
                def get_version_number(filename):
                    if 'best' in filename:
                        return 999  # Ensure 'best' is at the end
                    try:
                        # Extract number after 'v1_'
                        version = int(filename.split('_')[1].split('.')[0])
                        return version
                    except:
                        return 0
                
                # Sort by version number
                sorted_versions = sorted(model_versions, key=get_version_number)
                version_numbers = [get_version_number(v) for v in sorted_versions]
                
                if 999 in version_numbers:  # If 'best' version exists
                    default_idx = version_numbers.index(999)
                else:
                    default_idx = len(version_numbers) - 1  # Select the highest version
                
                st.markdown('<div class="slider-container">', unsafe_allow_html=True)
                st.markdown('<label class="select-label">Model Version</label>', unsafe_allow_html=True)
                
                # Create a list of labels for display
                slider_labels = {i: sorted_versions[i] for i in range(len(sorted_versions))}
                
                # Use selectbox instead of slider since it supports format_func
                selected_idx = st.selectbox(
                    "Model Version",
                    options=list(range(len(sorted_versions))),
                    format_func=lambda i: sorted_versions[i],
                    index=default_idx,
                    label_visibility="collapsed"
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
                model_version = sorted_versions[selected_idx]
        
        # Apply model button with pulsing effect
        st.markdown('<div class="apply-model-container">', unsafe_allow_html=True)
        if st.button("💫 Apply Selected Model", key="apply_model", use_container_width=True):
            st.session_state["selected_model"] = model_type
            st.session_state["selected_model_version"] = model_version
            # st.rerun()
            st.success(f"Applied model: {model_type} ({model_version})")

        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close model-selection-card
        
        # Display currently used model in a badge-style display
        current_model = st.session_state.get("selected_model", "CNN1D")
        current_version = st.session_state.get("selected_model_version", "Default")
        
        st.markdown("""
        <div class="active-model-container">
            <div class="active-model-badge">
                <div class="active-model-title">Active Model</div>
                <div class="active-model-details">
                    <span class="model-type-badge">""" + str(current_model) + """</span>
                    <span class="model-version-badge">""" + str(current_version) + """</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display prediction results AFTER model selection
        st.markdown('<h2 class="result-subheader with-icon"><span class="subheader-icon">👃</span> Smell Analysis Results</h2>', unsafe_allow_html=True)
        
        # In a real scenario, we would load and use the model here
        # For this example, we'll simulate the prediction results
        smell_options = ["Almond", "Anise", "Apricot", "Bael", "Beef", "Bergamot", "Black tea"]
        predicted_label = random.choice(smell_options)
        confidence = random.uniform(70, 98)
        
        # Display prediction result with animated reveal
        st.markdown(f"""
        <div class="prediction-container glass-card">
            <div class="prediction-header">
                <div class="prediction-icon">🎯</div>
                <h2>Predicted Smell</h2>
            </div>
            <div class="result-label animated-gradient">{str(predicted_label)}</div>
            <div class="confidence-meter">
                <div class="confidence-fill" style="width: {confidence:.1f}%;"></div>
            </div>
            <div class="confidence-value">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show Top 3 Predictions
        st.markdown('<h3 class="result-subheader with-icon"><span class="subheader-icon">🏆</span> Top Predictions</h3>', unsafe_allow_html=True)
        
        top_predictions = [
            {"label": predicted_label, "probability": confidence},
            {"label": smell_options[(smell_options.index(predicted_label) + 1) % len(smell_options)], 
             "probability": random.uniform(30, 60)},
            {"label": smell_options[(smell_options.index(predicted_label) + 2) % len(smell_options)], 
             "probability": random.uniform(10, 30)}
        ]
        
        # Use the imported function for plotting prediction results
        plot_prediction_results(top_predictions)
        
        # # I will update soon
        
        # Add export options
        # st.markdown('<div class="export-section">', unsafe_allow_html=True)
        # st.markdown('<h3 class="result-subheader with-icon"><span class="subheader-icon">📤</span> Export Results</h3>', unsafe_allow_html=True)
        
        # export_cols = st.columns(3)
        # with export_cols[0]:
        #     st.download_button(
        #         label="📊 Export as CSV",
        #         data="dummy data",  # In real app, generate CSV of results
        #         file_name=f"enose_results_{cycle_loop}.csv",
        #         mime="text/csv",
        #         use_container_width=True
        #     )
        # with export_cols[1]:
        #     st.download_button(
        #         label="📑 Export as PDF",
        #         data="dummy data",  # In real app, generate PDF report
        #         file_name=f"enose_report_{cycle_loop}.pdf",
        #         mime="application/pdf",
        #         use_container_width=True
        #     )
        # with export_cols[2]:
        #     st.download_button(
        #         label="📋 Export as JSON",
        #         data="dummy data",  # In real app, generate JSON of results
        #         file_name=f"enose_data_{cycle_loop}.json",
        #         mime="application/json",
        #         use_container_width=True
        #     )
        # st.markdown('</div>', unsafe_allow_html=True)
            
    # Case 2: Direct access to Result page
    else:
        MODEL_OPTIONS = {
            "CNN1D": "1D Convolutional Neural Network",
            "LSTMNet": "Long Short-Term Memory Network",
            "TransformerNet": "Transformer Neural Network"
        }
        
        # "No Analysis Data Available" message in a nice info box
        st.markdown("""
        <div class="no-data-container glass-card">
            <div class="no-data-icon">ℹ️</div>
            <h3>No Analysis Data Available</h3>
            <p>To analyze your E-nose data, please follow the steps below:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a modern workflow steps display
        st.markdown('<div class="workflow-container">', unsafe_allow_html=True)
        
        cols = st.columns(4)
        
        steps = [
            {"icon": "📤", "title": "Upload Data", "desc": "Upload your E-nose dataset CSV file"},
            {"icon": "🧹", "title": "Preprocess", "desc": "Clean and normalize your data"},
            {"icon": "📊", "title": "Analyze", "desc": "Explore your dataset visually"},
            {"icon": "📨", "title": "Submit", "desc": "Process with ML model"}
        ]
        
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"""
                <div class="workflow-step">
                    <div class="step-number">{i+1}</div>
                    <div class="step-icon">{steps[i]["icon"]}</div>
                    <h3 class="step-title">{steps[i]["title"]}</h3>
                    <p class="step-desc">{steps[i]["desc"]}</p>
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Separator with animated line
        st.markdown('<div class="animated-separator"></div>', unsafe_allow_html=True)
        
        # Available models section with modern cards
        st.markdown('<h2 class="section-header with-icon"><span class="section-icon">🧠</span> Available Analysis Models</h2>', unsafe_allow_html=True)
        
        # Show available models in a grid
        available_models = get_available_models()
        st.markdown('<div class="model-cards-container">', unsafe_allow_html=True)
        
        cols = st.columns(3)
        
        model_icons = {
            "CNN1D": "📊",
            "LSTMNet": "🔄", 
            "TransformerNet": "⚡"
        }
        
        for i, (model_key, model_name) in enumerate(MODEL_OPTIONS.items()):
            with cols[i]:
                model_versions = available_models.get(model_key, [])
                num_versions = len(model_versions)
                has_best = any('best' in v for v in model_versions)
                
                # Create visually appealing model cards
                st.markdown(f"""
                <div class="model-card glass-card">
                    <div class="model-icon">{model_icons[model_key]}</div>
                    <h3 class="model-title">{model_name}</h3>
                    <div class="model-type">{model_key}</div>
                    <div class="model-divider"></div>
                """, unsafe_allow_html=True)
                
                # Display status with color-coded method
                if num_versions > 0:
                    st.markdown(f"""
                    <div class="model-status success">
                        <span class="status-dot success-dot"></span>
                        {num_versions} versions available
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if has_best:
                        st.markdown("""
                        <div class="model-badge">
                            <span class="badge-icon">✓</span>
                            Best model available
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="model-status warning">
                        <span class="status-dot warning-dot"></span>
                        No models found
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown('</div>', unsafe_allow_html=True)  # Close model-card
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close model-cards-container
        
        # Add a tip section at the bottom with modern card design
        st.markdown('<div class="animated-separator"></div>', unsafe_allow_html=True)
        
        with st.expander("💡 How to get started"):
            st.markdown("""
            <div class="tip-container">
                <div class="tip-steps">
                    <div class="tip-step">
                        <div class="tip-step-number">1</div>
                        <div class="tip-step-content">
                            Go to the <span class="highlight">Process</span> tab using the navigation menu on the left
                        </div>
                    </div>
                    <div class="tip-step">
                        <div class="tip-step-number">2</div>
                        <div class="tip-step-content">
                            Upload your E-nose data file <span class="highlight">(.csv format)</span>
                        </div>
                    </div>
                    <div class="tip-step">
                        <div class="tip-step-number">3</div>
                        <div class="tip-step-content">
                            Select a cycle to analyze
                        </div>
                    </div>
                    <div class="tip-step">
                        <div class="tip-step-number">4</div>
                        <div class="tip-step-content">
                            Click <span class="highlight">"Submit and Save Analysis"</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)