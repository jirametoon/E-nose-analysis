"""
Home page implementation for the E-nose Analytics application.
"""
import streamlit as st
from .styles import get_home_styles

def render():
    """
    Renders the home page with feature overview and getting started section.
    """
    # Apply custom CSS
    st.markdown(get_home_styles(), unsafe_allow_html=True)
    
    # Main container
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    # 1) Enhanced Header with animated gradient
    st.markdown(
        '<div class="app-header">'
        '<h1 class="app-title">🔬 E-nose Analytics</h1>'
        '<p class="app-subtitle">Welcome to our specialized platform for electronic nose data visualization and analysis. '
        'Upload your sensor data, process and visualize odor patterns, and gain valuable insights into complex sensor readings.</p>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # 2) Feature Cards with glass card effect and animations
    features = [
        {
            "icon": "📊", 
            "title": "Sensor Data Management",
            "desc": "Easily upload CSV files containing electronic nose sensor measurements. Our system automatically processes and organizes the G-column sensor data for seamless analysis."
        },
        {
            "icon": "⚙️", 
            "title": "Signal Processing",
            "desc": "Clean, normalize, and prepare sensor data through our automated pipeline. Handle cycle detection, remove noise, and standardize signals for consistent analysis."
        },
        {
            "icon": "📈", 
            "title": "Interactive Visualization",
            "desc": "Explore sensor patterns through dynamic charts and visual representations. Compare signals across different cycles and sensor types to identify distinctive odor signatures."
        },
        {
            "icon": "🔍", 
            "title": "Sensor Response Analysis",
            "desc": "Examine how individual sensors (G1-G15) respond to different odors. Identify key patterns and characteristic responses that define specific scents."
        },
        {
            "icon": "🧪", 
            "title": "Odor Signal Analytics",
            "desc": "Analyze odor profiles through comprehensive signal statistics. Review data patterns that reveal the unique composition and properties of different scents."
        },
        {
            "icon": "📋", 
            "title": "Intuitive Interface",
            "desc": "Navigate through the entire data analysis workflow with our user-friendly interface. No technical expertise required to gain insights from complex sensor data."
        },
    ]
    
    # Feature grid rendering with glass card effect
    st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
    
    for i, f in enumerate(features):
        # Add a small delay to each card for staggered animation
        delay = i * 0.1
        st.markdown(f"""
            <div class="feature-card" style="animation: fadeIn 0.8s ease-in-out {delay}s both;">
                <div class="feature-icon">{f['icon']}</div>
                <h3 class="feature-title">{f['title']}</h3>
                <p class="feature-desc">{f['desc']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 3) Enhanced Next Steps section with modern styling
    st.markdown("""
        <div class="next-steps">
            <h2 class="next-steps-title with-icon">
                <span class="section-icon">🚀</span>
                Get Started in Minutes
            </h2>
            <ul class="step-list">
                <li class="step-item" style="animation-delay: 0.1s">
                    <div class="step-number">1</div>
                    <div class="step-text">Go to <strong class="highlight">Process</strong> tab to upload your E-nose CSV files and view sensor readings</div>
                </li>
                <li class="step-item" style="animation-delay: 0.3s">
                    <div class="step-number">2</div>
                    <div class="step-text">Switch to <strong class="highlight">Results</strong> to see the visualization and analysis of your odor data</div>
                </li>
            </ul>
    """, unsafe_allow_html=True)
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)