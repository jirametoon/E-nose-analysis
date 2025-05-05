"""
Process page implementation for the E-nose Analytics application.
"""
import streamlit as st
import pandas as pd
import io

from ..pipeline_streamlit import handle_pipeline
from ..viz_utils import plot_multiple_graphs
from .styles import get_process_styles

def render():
    """
    Renders the process page for data uploading and preprocessing.
    """
    # Apply custom CSS
    st.markdown(get_process_styles(), unsafe_allow_html=True)
    
    # Main container
    st.markdown('<div class="process-container">', unsafe_allow_html=True)
    
    # Header with icon
    st.markdown(f"""
    <h1 class="process-header">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15Z" stroke="#3a7bd5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M19.4 15C19.2669 15.3016 19.2272 15.6362 19.286 15.9606C19.3448 16.285 19.4995 16.5843 19.73 16.82L19.79 16.88C19.976 17.0657 20.1235 17.2863 20.2241 17.5291C20.3248 17.7719 20.3766 18.0322 20.3766 18.295C20.3766 18.5578 20.3248 18.8181 20.2241 19.0609C20.1235 19.3037 19.976 19.5243 19.79 19.71C19.6043 19.896 19.3837 20.0435 19.1409 20.1441C18.8981 20.2448 18.6378 20.2966 18.375 20.2966C18.1122 20.2966 17.8519 20.2448 17.6091 20.1441C17.3663 20.0435 17.1457 19.896 16.96 19.71L16.9 19.65C16.6643 19.4195 16.365 19.2648 16.0406 19.206C15.7162 19.1472 15.3816 19.1869 15.08 19.32C14.7842 19.4468 14.532 19.6572 14.3543 19.9255C14.1766 20.1938 14.0813 20.5082 14.08 20.83V21C14.08 21.5304 13.8693 22.0391 13.4942 22.4142C13.1191 22.7893 12.6104 23 12.08 23C11.5496 23 11.0409 22.7893 10.6658 22.4142C10.2907 22.0391 10.08 21.5304 10.08 21V20.91C10.0723 20.579 9.96512 20.258 9.77251 19.9887C9.5799 19.7194 9.31074 19.5143 9 19.4C8.69838 19.2669 8.36381 19.2272 8.03941 19.286C7.71502 19.3448 7.41568 19.4995 7.18 19.73L7.12 19.79C6.93425 19.976 6.71368 20.1235 6.47088 20.2241C6.22808 20.3248 5.96783 20.3766 5.705 20.3766C5.44217 20.3766 5.18192 20.3248 4.93912 20.2241C4.69632 20.1235 4.47575 19.976 4.29 19.79C4.10405 19.6043 3.95653 19.3837 3.85588 19.1409C3.75523 18.8981 3.70343 18.6378 3.70343 18.375C3.70343 18.1122 3.75523 17.8519 3.85588 17.6091C3.95653 17.3663 4.10405 17.1457 4.29 16.96L4.35 16.9C4.58054 16.6643 4.73519 16.365 4.794 16.0406C4.85282 15.7162 4.81312 15.3816 4.68 15.08C4.55324 14.7842 4.34276 14.532 4.07447 14.3543C3.80618 14.1766 3.49179 14.0813 3.17 14.08H3C2.46957 14.08 1.96086 13.8693 1.58579 13.4942C1.21071 13.1191 1 12.6104 1 12.08C1 11.5496 1.21071 11.0409 1.58579 10.6658C1.96086 10.2907 2.46957 10.08 3 10.08H3.09C3.42099 10.0723 3.742 9.96512 4.0113 9.77251C4.28059 9.5799 4.48572 9.31074 4.6 9C4.73312 8.69838 4.77282 8.36381 4.714 8.03941C4.65519 7.71502 4.50054 7.41568 4.27 7.18L4.21 7.12C4.02405 6.93425 3.87653 6.71368 3.77588 6.47088C3.67523 6.22808 3.62343 5.96783 3.62343 5.705C3.62343 5.44217 3.67523 5.18192 3.77588 4.93912C3.87653 4.69632 4.02405 4.47575 4.21 4.29C4.39575 4.10405 4.61632 3.95653 4.85912 3.85588C5.10192 3.75523 5.36217 3.70343 5.625 3.70343C5.88783 3.70343 6.14808 3.75523 6.39088 3.85588C6.63368 3.95653 6.85425 4.10405 7.04 4.29L7.1 4.35C7.33568 4.58054 7.63502 4.73519 7.95941 4.794C8.28381 4.85282 8.61838 4.81312 8.92 4.68H9C9.29577 4.55324 9.54802 4.34276 9.72569 4.07447C9.90337 3.80618 9.99872 3.49179 10 3.17V3C10 2.46957 10.2107 1.96086 10.5858 1.58579C10.9609 1.21071 11.4696 1 12 1C12.5304 1 13.0391 1.21071 13.4142 1.58579C13.7893 1.96086 14 2.46957 14 3V3.09C14.0013 3.41179 14.0966 3.72618 14.2743 3.99447C14.452 4.26276 14.7042 4.47324 15 4.6C15.3016 4.73312 15.6362 4.77282 15.9606 4.714C16.285 4.65519 16.5843 4.50054 16.82 4.27L16.88 4.21C17.0657 4.02405 17.2863 3.87653 17.5291 3.77588C17.7719 3.67523 18.0322 3.62343 18.295 3.62343C18.5578 3.62343 18.8181 3.67523 19.0609 3.77588C19.3037 3.87653 19.5243 4.02405 19.71 4.21C19.896 4.39575 20.0435 4.61632 20.1441 4.85912C20.2448 5.10192 20.2966 5.36217 20.2966 5.625C20.2966 5.88783 20.2448 6.14808 20.1441 6.39088C20.0435 6.63368 19.896 6.85425 19.71 7.04L19.65 7.1C19.4195 7.33568 19.2648 7.63502 19.206 7.95941C19.1472 8.28381 19.1869 8.61838 19.32 8.92V9C19.4468 9.29577 19.6572 9.54802 19.9255 9.72569C20.1938 9.90337 20.5082 9.99872 20.83 10H21C21.5304 10 22.0391 10.2107 22.4142 10.5858C22.7893 10.9609 23 11.4696 23 12C23 12.5304 22.7893 13.0391 22.4142 13.4142C22.0391 13.7893 21.5304 14 21 14H20.91C20.5882 14.0013 20.2738 14.0966 20.0055 14.2743C19.7372 14.452 19.5268 14.7042 19.4 15Z" stroke="#3a7bd5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        Data Processing Pipeline
    </h1>
    """, unsafe_allow_html=True)
    
    # Process steps - Updated to remove model selection step
    st.markdown("""
    <div class="step-container">
        <div class="step">
            <div class="step-number">1</div>
            <div class="step-title">Upload Data</div>
        </div>
        <div class="step">
            <div class="step-line"></div>
            <div class="step-number">2</div>
            <div class="step-title">Preprocess</div>
        </div>
        <div class="step">
            <div class="step-line"></div>
            <div class="step-number">3</div>
            <div class="step-title">Analyze</div>
        </div>
        <div class="step">
            <div class="step-line"></div>
            <div class="step-number">4</div>
            <div class="step-title">Submit to Model</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 1: Upload section
    # Check if file already uploaded
    already_uploaded = "uploaded_bytes" in st.session_state
    
    if not already_uploaded:
        st.markdown("""
        <div class="upload-area">
            <div class="upload-icon">📁</div>
            <div class="upload-text">Drag and drop your CSV file here or click to browse</div>
        </div>
        """, unsafe_allow_html=True)
    
    # File uploader
    uploaded = st.file_uploader("Upload your CSV file", type="csv", key="uploaded_widget")
    if uploaded:
        st.session_state["uploaded_bytes"] = uploaded.read()
        st.session_state["uploaded_name"] = uploaded.name
    
    # Return early if no file uploaded
    if "uploaded_bytes" not in st.session_state:
        st.info("Waiting for CSV file upload...")
        st.markdown('</div>', unsafe_allow_html=True)  # Close main container
        return
    
    # Step 2: Raw data preview
    st.markdown(f"""
    <h2 class="process-subheader">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="#3a7bd5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M14 2V8H20" stroke="#3a7bd5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M16 13H8" stroke="#3a7bd5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M16 17H8" stroke="#3a7bd5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M10 9H9H8" stroke="#3a7bd5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        Raw Data Preview: {st.session_state['uploaded_name']}
    </h2>
    """, unsafe_allow_html=True)
    
    # Load and preview raw data
    df_raw = pd.read_csv(io.BytesIO(st.session_state["uploaded_bytes"]))
    
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(df_raw)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 3: Preprocess data
    st.markdown(f"""
    <h2 class="process-subheader">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 8V16" stroke="#3a7bd5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M8 12H16" stroke="#3a7bd5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M10 2H14" stroke="#3a7bd5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M21 15C21 16.8565 20.2625 18.637 18.9497 19.9497C17.637
             21.2625 15.8565 22 14 22H10C8.14348 22 6.36301 21.2625 5.05025 19.9497C3.7375 18.637 3 15.8565 3 15V9C3 7.14348 3.7375 5.36301 5.05025 4.05025C6.36301 2.7375 8.14348 2 10 2H14C15.8565 2 17.637 2.7375 18.9497 4.05025C20.2625 5.36301 21 7.14348 21 9V15Z" stroke="#3a7bd5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        Processed Data Preview
    </h2>
    """, unsafe_allow_html=True)
    
    # Process data
    df_processed = handle_pipeline(df_raw)
    
    # Store processed data in session state
    if "df_processed" not in st.session_state:
        st.session_state["df_processed"] = df_processed
    
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(df_processed)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 4: Plot distribution wrapped in a card
    st.markdown(f"""
    <h2 class="section-header">
        <div class="icon-circle" style="background: linear-gradient(135deg, #3a7bd5, #00d2ff); width: 42px; height: 42px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(58, 123, 213, 0.3);">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M21 21H3.8C3.58783 21 3.38434 20.9157 3.23431 20.7657C3.08429 20.6157 3 20.4122 3 20.2V3" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M7 14.2L11.3 9.9L15.5 14.1L20 9.6" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        G1-G15 Sensor Data Visualization
    </h2>
    """, unsafe_allow_html=True)
    
    # Replace the old cycle selector with this new beautiful version
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(58, 123, 213, 0.1), rgba(0, 210, 255, 0.1)); 
                border-radius: var(--radius-md); 
                padding: 1.75rem; 
                margin: 1.5rem 0; 
                border: 1px solid rgba(58, 123, 213, 0.2);
                box-shadow: 0 8px 20px rgba(58, 123, 213, 0.05);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;">
        <div style="position: absolute; 
                    top: 0; right: 0; 
                    width: 150px; height: 150px; 
                    background: radial-gradient(circle at top right, rgba(0, 210, 255, 0.3), transparent 70%);
                    border-radius: 0 0 0 100%;
                    z-index: 0;"></div>
        <div style="position: absolute; 
                    bottom: 0; left: 0; 
                    width: 100px; height: 100px; 
                    background: radial-gradient(circle at bottom left, rgba(58, 123, 213, 0.2), transparent 70%);
                    border-radius: 0 100% 0 0;
                    z-index: 0;"></div>
        <div style="position: relative; z-index: 1;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="background: linear-gradient(135deg, #3a7bd5, #00d2ff); 
                            width: 40px; height: 40px; 
                            border-radius: 50%; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center;
                            margin-right: 15px;
                            box-shadow: 0 4px 10px rgba(58, 123, 213, 0.3);">
                    <span style="font-size: 20px; color: white;">📊</span>
                </div>
                <h3 style="margin: 0; font-size: 1.3rem; font-weight: 600; 
                           background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
                           -webkit-background-clip: text;
                           -webkit-text-fill-color: transparent;">
                    Select Cycle Loop for Analysis
                </h3>
            </div>
            <p style="color: var(--text-secondary); 
                      margin-bottom: 1rem; 
                      padding-left: 55px;
                      font-size: 0.95rem;">
                Choose a cycle loop to visualize the sensor response patterns for detailed analysis
            </p>
    """, unsafe_allow_html=True)
    
    # Cycle selection - Now inside our beautifully styled container
    cycle_options = sorted(df_processed["Cycle Loop"].unique())
    selected_cycle = st.selectbox("Select Cycle Loop", cycle_options)
    
    # Store selected cycle in session state and close div
    st.session_state["selected_cycle"] = selected_cycle
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Plot graphs
    plot_multiple_graphs(df_processed, [f"G{i}" for i in range(1,16)], cycle=selected_cycle)
    st.markdown('</div>', unsafe_allow_html=True)  # Close the card
    
    # Submit section with enhanced styling matching the cycle selector
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(58, 123, 213, 0.1), rgba(0, 210, 255, 0.1)); 
                border-radius: var(--radius-md); 
                padding: 1.75rem; 
                margin: 1.5rem 0; 
                border: 1px solid rgba(58, 123, 213, 0.2);
                box-shadow: 0 8px 20px rgba(58, 123, 213, 0.05);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;">
        <div style="position: absolute; 
                    top: 0; right: 0; 
                    width: 150px; height: 150px; 
                    background: radial-gradient(circle at top right, rgba(0, 210, 255, 0.3), transparent 70%);
                    border-radius: 0 0 0 100%;
                    z-index: 0;"></div>
        <div style="position: absolute; 
                    bottom: 0; left: 0; 
                    width: 100px; height: 100px; 
                    background: radial-gradient(circle at bottom left, rgba(58, 123, 213, 0.2), transparent 70%);
                    border-radius: 0 100% 0 0;
                    z-index: 0;"></div>
        <div style="position: relative; z-index: 1;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="background: linear-gradient(135deg, #3a7bd5, #00d2ff); 
                            width: 40px; height: 40px; 
                            border-radius: 50%; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center;
                            margin-right: 15px;
                            box-shadow: 0 4px 10px rgba(58, 123, 213, 0.3);">
                    <span style="font-size: 20px; color: white;">📋</span>
                </div>
                <h3 style="margin: 0; font-size: 1.3rem; font-weight: 600; 
                           background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
                           -webkit-background-clip: text;
                           -webkit-text-fill-color: transparent;">
                    Submit Data for Analysis
                </h3>
            </div>
            <p style="color: var(--text-secondary); 
                      margin-bottom: 1.5rem; 
                      padding-left: 55px;
                      font-size: 0.95rem;">
                Your data is processed and ready for detailed analysis. Click the button below to save your analysis.
            </p>
    """, unsafe_allow_html=True)
    
    # Submit button with improved styling
    if st.button("📊 Submit and Save Analysis", key="save_analysis", use_container_width=True):
        # Set default model (since we removed model selection)
        st.session_state["selected_model"] = "CNN1D"
        
        # Save data needed for result page
        st.session_state["analysis_ready"] = True
        st.session_state["selected_cycle"] = selected_cycle
        
        # Just show a success message without redirecting
        st.toast("✅ Data is ready! Go to Results tab to view your analysis.", icon="✅")
        
        # Show success message with nice styling
        st.markdown("""
        <div style="background: linear-gradient(135deg, #12c2e9, #c471ed, #f64f59);
                    color: white;
                    border-radius: var(--radius-md);
                    padding: 1.25rem;
                    margin-top: 1rem;
                    box-shadow: 0 8px 20px rgba(196, 113, 237, 0.15);
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    animation: fadeIn 0.5s ease-out;">
            <div style="background: rgba(255,255,255,0.2);
                        width: 40px; height: 40px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        flex-shrink: 0;">
                <span style="font-size: 20px;">✅</span>
            </div>
            <div>
                <h4 style="margin: 0 0 0.5rem 0; font-size: 1.2rem;">Analysis Complete!</h4>
                <p style="margin: 0; font-size: 0.95rem;">Your data is ready for viewing. Please navigate to the Results tab to see your analysis.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Close submit section div
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)