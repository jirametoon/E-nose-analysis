"""
Info page implementation for the E-nose Analytics application.
"""
import streamlit as st
from .styles import get_info_styles

def render():
    """
    Renders the information page with explanations about E-nose technology.
    """
    # Apply custom CSS
    st.markdown(get_info_styles(), unsafe_allow_html=True)
    
    # Main container
    st.markdown('<div class="info-container">', unsafe_allow_html=True)
    
    # Introduction section with logo and description - Using Streamlit columns
    st.markdown('<h2 class="info-subheader">What is an E-nose?</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Use Streamlit's native image function
        st.image("assets/logo.png", use_container_width=True)
    
    with col2:
        # Using HTML for styled text
        st.markdown("""
        <div class="description-box">
            <p><strong>E-nose</strong> is a device that mimics the human sense of smell.</p>
            <p>It uses an array of sensors to detect volatile compounds in the air and analyze odors.</p>
            <p>The sensors generate signals, which are processed by pattern recognition software to identify specific smells.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sensor explanation section
    st.markdown('<h2 class="info-subheader">How the Sensors Work</h2>', unsafe_allow_html=True)
    
    # Using Streamlit columns for sensor images
    col1, col2 = st.columns(2)
    
    with col1:
        # Use Streamlit's native image function with caption
        st.image("assets/equation_air_state.png", use_container_width=True)
        st.image("assets/sensor_air_state.png", caption="Air state", use_container_width=True)
    
    with col2:
        # Use Streamlit's native image function with caption
        st.image("assets/equation_odor_state.png", use_container_width=True)
        st.image("assets/sensor_odor_state.png", caption="Odor state", use_container_width=True)
    
    # Explanation text
    st.markdown("""
    <div class="info-highlight">
        <p style="margin: 0; font-size: 1.1rem; line-height: 1.7;">
            In the <strong>air state</strong>, oxygen molecules capture electrons, resulting in low electrical conductivity. When <strong>odor molecules</strong> enter and bind with the oxygen, electrons are released back into the system, significantly increasing the sensor's electrical conductivity. This change in conductivity serves as the basis for odor detection and analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # E-nose diagram section
    st.markdown('<h2 class="info-header">Electronic Nose System</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="info-text">
        The complete electronic nose system integrates multiple sensors with advanced data processing to accurately identify and classify different odors. The diagram below illustrates the full workflow from odor sampling to pattern recognition.
    </p>
    """, unsafe_allow_html=True)
    
    # Use Streamlit's native image function for the diagram
    st.image("assets/e-nose_diagram.png", use_container_width=True)
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)