"""
Centralized styles for the application UI components.
This module contains CSS style definitions used across all pages.
"""

# Common CSS color variables and other constants
COLOR_PRIMARY = "#3a7bd5"
COLOR_SECONDARY = "#00d2ff"
COLOR_TEXT_PRIMARY = "#202124"
COLOR_TEXT_SECONDARY = "#5f6368"
COLOR_BG_LIGHT = "#f5f7fa"
COLOR_BG_LIGHTER = "#ffffff"
COLOR_BORDER_LIGHT = "rgba(0,0,0,0.05)"
COLOR_BORDER_PRIMARY = f"rgba(58, 123, 213, 0.2)"
GRADIENT_PRIMARY = f"linear-gradient(135deg, {COLOR_PRIMARY}, {COLOR_SECONDARY})"
GRADIENT_SECONDARY = f"linear-gradient(135deg, {COLOR_BG_LIGHT} 0%, #e4ecfb 100%)"
FONT_FAMILY_PRIMARY = "'Poppins', sans-serif"
FONT_FAMILY_DISPLAY = "'Playfair Display', serif"


def get_base_styles():
    """
    Returns base CSS styles used across all pages.
    This contains common elements like fonts, animations, and base component styles.
    """
    return f"""
    /* Base styles and animations */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');
    
    :root {{
        --primary-color: {COLOR_PRIMARY};
        --secondary-color: {COLOR_SECONDARY};
        --text-primary: {COLOR_TEXT_PRIMARY};
        --text-secondary: {COLOR_TEXT_SECONDARY};
        --bg-light: {COLOR_BG_LIGHT};
        --bg-lighter: {COLOR_BG_LIGHTER};
        --border-light: {COLOR_BORDER_LIGHT};
        --border-primary: {COLOR_BORDER_PRIMARY};
        --shadow-sm: 0 4px 20px rgba(0,0,0,0.05);
        --shadow-md: 0 8px 30px rgba(0,0,0,0.08);
        --shadow-lg: 0 12px 40px rgba(0,0,0,0.1);
        --gradient-primary: {GRADIENT_PRIMARY};
        --gradient-secondary: {GRADIENT_SECONDARY};
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --transition-standard: all 0.3s ease;
    }}
    
    body {{
        font-family: {FONT_FAMILY_PRIMARY};
        color: var(--text-primary);
        line-height: 1.6;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        font-family: {FONT_FAMILY_PRIMARY};
        margin-top: 0;
    }}
    
    p {{
        margin-top: 0;
    }}
    
    /* Common Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes slideIn {{
        from {{ opacity: 0; transform: translateX(-20px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}
    
    @keyframes gradient {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    """


def get_component_styles():
    """
    Returns common component styles used across multiple pages.
    """
    return f"""
    /* Common Components */
    .container {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 1.5rem;
        animation: fadeIn 0.8s ease-in-out;
    }}
    
    /* Header styling */
    .page-header {{
        margin-bottom: 2rem;
        border-bottom: 2px solid rgba(49, 51, 63, 0.1);
        padding-bottom: 1.5rem;
    }}
    
    .page-title {{
        font-family: {FONT_FAMILY_DISPLAY};
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.1);
    }}
    
    .page-subtitle {{
        font-weight: 300;
        font-size: 1.25rem;
        color: var(--text-secondary);
        line-height: 1.6;
        max-width: 800px;
    }}
    
    /* Card styling */
    .card {{
        background: var(--bg-lighter);
        border-radius: var(--radius-md);
        padding: 1.75rem;
        transition: var(--transition-standard);
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-light);
        margin-bottom: 1.5rem;
    }}
    
    .card:hover {{
        transform: translateY(-5px);
        box-shadow: var(--shadow-md);
        border-color: var(--border-primary);
    }}
    
    /* Highlight sections */
    .highlight-section {{
        background: var(--gradient-secondary);
        border-radius: var(--radius-md);
        padding: 1.75rem;
        margin: 1.5rem 0;
        border: 1px solid var(--border-primary);
    }}
    
    /* Button styling */
    .btn {{
        background: var(--gradient-primary);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: var(--radius-sm);
        font-weight: 600;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        transition: var(--transition-standard);
        box-shadow: 0 4px 10px rgba(58, 123, 213, 0.3);
        border: none;
        cursor: pointer;
        font-family: {FONT_FAMILY_PRIMARY};
        font-size: 1rem;
    }}
    
    .btn:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(58, 123, 213, 0.4);
    }}
    
    /* Divider */
    .divider {{
        height: 2px;
        background: linear-gradient(90deg, rgba(58, 123, 213, 0.1), rgba(0, 210, 255, 0.5), rgba(58, 123, 213, 0.1));
        margin: 2.5rem 0;
        border-radius: 2px;
    }}
    
    /* Section headers */
    .section-header {{
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0 1.5rem 0;
        color: var(--text-primary);
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}
    
    /* Icons */
    .icon {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}
    
    .icon-circle {{
        background: var(--gradient-primary);
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    /* Streamlit overrides */
    .stButton > button {{
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: var(--radius-sm);
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: var(--transition-standard);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(58, 123, 213, 0.25);
    }}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {{
        .page-title {{
            font-size: 2.5rem;
        }}
    }}
    """


def get_common_styles():
    """
    Returns common CSS styles used across all pages wrapped in style tags.
    """
    return f"<style>{get_base_styles()}{get_component_styles()}</style>"


def get_home_styles():
    """
    Returns CSS styles for the home page wrapped in style tags.
    """
    return f"""<style>
    {get_base_styles()}
    {get_component_styles()}
    
    .container {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
        animation: fadeIn 0.8s ease-in-out;
    }}
    
    /* Header styling */
    .app-header {{
        margin-bottom: 2rem;
        border-bottom: 2px solid rgba(49, 51, 63, 0.1);
        padding-bottom: 1.5rem;
    }}
    
    .app-title {{
        font-family: {FONT_FAMILY_DISPLAY};
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, {COLOR_PRIMARY}, {COLOR_SECONDARY});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.1);
    }}
    
    .app-subtitle {{
        font-family: {FONT_FAMILY_PRIMARY};
        font-size: 1.25rem;
        font-weight: 300;
        color: {COLOR_TEXT_SECONDARY};
        line-height: 1.6;
        max-width: 800px;
    }}
    
    /* Feature grid styling */
    .feature-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }}
    
    .feature-card {{
        background: white;
        border-radius: 12px;
        padding: 1.75rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 30px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
    }}
    
    .feature-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.1);
        border-color: rgba(58, 123, 213, 0.3);
    }}
    
    .feature-icon {{
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: inline-block;
    }}
    
    .feature-title {{
        font-family: {FONT_FAMILY_PRIMARY};
        font-weight: 600;
        font-size: 1.4rem;
        margin-bottom: 0.8rem;
        color: {COLOR_TEXT_PRIMARY};
    }}
    
    .feature-desc {{
        font-family: {FONT_FAMILY_PRIMARY};
        font-weight: 300;
        line-height: 1.6;
        font-size: 1rem;
        color: {COLOR_TEXT_SECONDARY};
        flex-grow: 1;
    }}
    
    /* Next steps section */
    .next-steps {{
        background: {GRADIENT_SECONDARY};
        padding: 2rem;
        border-radius: 12px;
        margin-top: 2rem;
        border: 1px solid {COLOR_BORDER_PRIMARY};
    }}
    
    .next-steps-title {{
        font-family: {FONT_FAMILY_PRIMARY};
        font-weight: 600;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        color: {COLOR_TEXT_PRIMARY};
        display: flex;
        align-items: center;
    }}
    
    .next-steps-title svg {{
        margin-right: 0.5rem;
    }}
    
    .step-list {{
        font-family: {FONT_FAMILY_PRIMARY};
        padding-left: 0;
        list-style: none;
    }}
    
    .step-item {{
        display: flex;
        align-items: flex-start;
        padding: 0.75rem 0;
        animation: slideIn 0.5s ease-in-out;
        animation-fill-mode: both;
    }}
    
    .step-number {{
        background: {GRADIENT_PRIMARY};
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        margin-right: 1rem;
        flex-shrink: 0;
    }}
    
    .step-text {{
        font-size: 1.1rem;
        color: {COLOR_TEXT_SECONDARY};
        padding-top: 0.2rem;
    }}
    
    .step-text strong {{
        color: {COLOR_TEXT_PRIMARY};
        font-weight: 600;
    }}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {{
        .app-title {{
            font-size: 2.5rem;
        }}
        .feature-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    </style>"""


def get_info_styles():
    """
    Returns CSS styles for the info page wrapped in style tags.
    """
    return f"""<style>
    {get_base_styles()}
    {get_component_styles()}
    
    .info-container {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
        animation: fadeIn 0.8s ease-in-out;
        font-family: {FONT_FAMILY_PRIMARY};
    }}
    
    /* Section styling */
    .info-header {{
        font-family: {FONT_FAMILY_DISPLAY};
        font-size: 2.2rem;
        font-weight: 700;
        margin: 2rem 0 1.5rem 0;
        color: {COLOR_TEXT_PRIMARY};
        background: linear-gradient(90deg, {COLOR_PRIMARY}, {COLOR_SECONDARY});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .info-subheader {{
        font-family: {FONT_FAMILY_PRIMARY};
        font-weight: 600;
        font-size: 1.6rem;
        margin: 1.5rem 0 1rem 0;
        color: {COLOR_TEXT_PRIMARY};
    }}
    
    .info-text {{
        font-family: {FONT_FAMILY_PRIMARY};
        font-weight: 300;
        font-size: 1.1rem;
        line-height: 1.7;
        color: {COLOR_TEXT_SECONDARY};
        margin-bottom: 1.5rem;
    }}
    
    .info-highlight {{
        background: {GRADIENT_SECONDARY};
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border: 1px solid {COLOR_BORDER_PRIMARY};
    }}
    
    /* Logo and description section */
    .description-box {{
        background: linear-gradient(135deg, #ff7e5f, #feb47b);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        font-size: 1.25rem;
        line-height: 1.6;
        box-shadow: 0 8px 30px rgba(255, 126, 95, 0.3);
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }}
    
    .description-box p {{
        margin: 0.5rem 0;
    }}
    
    /* Sensor explanation section */
    .sensor-section {{
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
    }}
    
    .sensor-caption {{
        font-family: {FONT_FAMILY_PRIMARY};
        font-weight: 600;
        color: {COLOR_PRIMARY};
        font-size: 1.1rem;
        text-align: center;
        background: rgba(58, 123, 213, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 30px;
        margin-top: 0.5rem;
    }}
    
    /* Add image styling for Streamlit native images */
    .stImage {{
        border-radius: 12px !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1) !important;
        transition: transform 0.3s ease !important;
    }}
    
    .stImage:hover {{
        transform: scale(1.02) !important;
    }}
    
    .css-1kyxreq img {{  /* Targeting Streamlit image captions */
        margin-top: 0.5rem !important;
        color: {COLOR_PRIMARY} !important;
        font-family: {FONT_FAMILY_PRIMARY} !important;
        font-weight: 600 !important;
    }}
    </style>"""


def get_process_styles():
    """
    Returns CSS styles for the process page wrapped in style tags.
    """
    return f"""<style>
    {get_base_styles()}
    {get_component_styles()}
    
    .process-container {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
        animation: fadeIn 0.8s ease-in-out;
        font-family: {FONT_FAMILY_PRIMARY};
    }}
    
    /* Section styling */
    .process-header {{
        font-family: {FONT_FAMILY_DISPLAY};
        font-size: 2.5rem;
        font-weight: 700;
        margin: 1rem 0 2rem 0;
        color: {COLOR_TEXT_PRIMARY};
        background: linear-gradient(90deg, {COLOR_PRIMARY}, {COLOR_SECONDARY});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 1rem;
    }}
    
    .process-header svg {{
        flex-shrink: 0;
    }}
    
    .process-subheader {{
        font-family: {FONT_FAMILY_PRIMARY};
        font-weight: 600;
        font-size: 1.6rem;
        margin: 2rem 0 1rem 0;
        color: {COLOR_TEXT_PRIMARY};
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .process-card {{
        background: white;
        border-radius: 12px;
        padding: 1.75rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }}
    
    .process-card:hover {{
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        border-color: {COLOR_BORDER_PRIMARY};
    }}
    
    /* Upload section */
    .upload-area {{
        background: {GRADIENT_SECONDARY};
        border-radius: 12px;
        padding: 2rem;
        border: 2px dashed rgba(58, 123, 213, .4);
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 2rem;
    }}
    
    .upload-area:hover {{
        border-color: rgba(58, 123, 213, 0.8);
        background: linear-gradient(135deg, #f0f4fa 0%, #dce6fb 100%);
    }}
    
    .upload-icon {{
        font-size: 3rem;
        color: {COLOR_PRIMARY};
        margin-bottom: 1rem;
        animation: pulse 2s infinite ease-in-out;
    }}
    
    .upload-text {{
        font-family: {FONT_FAMILY_PRIMARY};
        font-weight: 300;
        font-size: 1.1rem;
        color: {COLOR_TEXT_SECONDARY};
        margin-bottom: 1rem;
    }}
    
    /* Dataframe styling */
    .dataframe-container {{
        margin: 1.5rem 0;
        overflow: hidden;
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }}
    
    /* Success message */
    .success-card {{
        background: linear-gradient(135deg, #3cba92, #0ba360);
        color: white;
        padding: 1.25rem;
        border-radius: 12px;
        font-size: 1.1rem;
        margin: 2rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }}
    
    .success-icon {{
        font-size: 1.5rem;
    }}
    
    /* Submit button */
    .submit-button {{
        background: {GRADIENT_PRIMARY};
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-family: {FONT_FAMILY_PRIMARY};
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(58, 123, 213, 0.3);
        margin-top: 1rem;
        width: 100%;
        text-align: center;
        justify-content: center;
    }}
    
    .submit-button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(58, 123, 213, 0.4);
    }}
    
    /* Cycle selector */
    .cycle-selector {{
        background: white;
        padding: 1.75rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }}
    
    .cycle-selector:hover {{
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        border-color: {COLOR_BORDER_PRIMARY};
    }}
    
    .cycle-label {{
        font-family: {FONT_FAMILY_PRIMARY};
        font-weight: 600;
        font-size: 1.2rem;
        color: {COLOR_TEXT_PRIMARY};
        margin-bottom: 1rem;
    }}
    
    /* Step indicator */
    .step-container {{
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
    }}
    
    .step {{
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
        position: relative;
    }}
    
    .step-number {{
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: {GRADIENT_PRIMARY};
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: {FONT_FAMILY_PRIMARY};
        font-weight: 600;
        margin-bottom: 0.75rem;
        position: relative;
        z-index: 2;
    }}
    
    .step-title {{
        font-family: {FONT_FAMILY_PRIMARY};
        font-weight: 600;
        font-size: 0.9rem;
        color: {COLOR_TEXT_PRIMARY};
        text-align: center;
    }}
    
    .step-line {{
        position: absolute;
        top: 20px;
        height: 2px;
        background: linear-gradient(90deg, rgba(58, 123, 213, 0.3), rgba(0, 210, 255, 0.3));
        width: 100%;
        left: -50%;
        z-index: 1;
    }}
    
    .step:first-child .step-line {{
        display: none;
    }}
    
    /* Override Streamlit default styling */
    .css-1kyxreq {{
        background-color: transparent !important;
        border: none !important;
    }}
    
    /* Add responsiveness */
    @media (max-width: 768px) {{
        .process-header {{
            font-size: 2rem;
        }}
        
        .step-container {{
            flex-direction: column;
            gap: 1.5rem;
        }}
        
        .step-line {{
            display: none;
        }}
    }}
    </style>"""


def get_result_styles():
    """
    Returns CSS styles for the results page wrapped in style tags.
    """
    return """<style>
        /* Base styling */
        .result-container {
            padding: 1.5rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Animations */
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(58, 123, 213, 0.4); }
            70% { box-shadow: 0 0 0 15px rgba(58, 123, 213, 0); }
            100% { box-shadow: 0 0 0 0 rgba(58, 123, 213, 0); }
        }
        
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes slideIn {
            0% { transform: translateX(-20px); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes fillUp {
            0% { width: 0%; }
            100% { width: 100%; }
        }
        
        /* Modern Cards - Glass effect */
        .glass-card {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }
        
        /* Headers */
        .header-wrapper {
            margin-bottom: 2rem;
        }
        
        .result-header {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            color: #3a7bd5;
            display: flex;
            align-items: center;
            padding: 0.5rem 0;
        }
        
        .header-icon {
            margin-right: 0.75rem;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #3a7bd5;
        }
        
        .animated-gradient {
            background: linear-gradient(90deg, #3a7bd5, #00d2ff, #3a7bd5);
            background-size: 300% 100%;
            -webkit-background-clip: text; 
            background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient 6s ease infinite;
        }
        
        .result-subheader {
            font-size: 1.5rem;
            font-weight: 600;
            margin: 1.5rem 0 1rem 0;
            color: #2c3e50;
            display: flex;
            align-items: center;
        }
        
        .with-icon {
            display: flex;
            align-items: center;
        }
        
        .subheader-icon, .section-icon {
            background: linear-gradient(135deg, #3a7bd5, #00d2ff);
            border-radius: 50%;
            width: 32px;
            height: 32px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-right: 0.75rem;
            color: white;
            font-size: 1rem;
        }
        
        .section-header {
            font-size: 2rem;
            font-weight: 700;
            margin: 2rem 0 1.5rem 0;
            color: #2c3e50;
        }
        
        /* Info panel styling */
        .info-panel {
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
        }
        
        .info-item {
            display: flex;
            align-items: center;
        }
        
        .info-icon {
            font-size: 1.5rem;
            margin-right: 1rem;
            background: linear-gradient(135deg, #3a7bd5, #00d2ff);
            border-radius: 50%;
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }
        
        .info-label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #64748b;
            margin-bottom: 0.25rem;
        }
        
        .info-value {
            font-size: 1.25rem;
            font-weight: 700;
            color: #1e293b;
        }
        
        /* Tabs styling */
        .custom-tabs-container {
            margin-bottom: 2rem;
        }
        
        .tab-content {
            padding: 1rem 0;
        }
        
        /* Model selection */
        .model-selection-card {
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .select-container, .slider-container {
            margin-bottom: 1rem;
        }
        
        .select-label {
            font-size: 0.9rem;
            font-weight: 600;
            color: #64748b;
            margin-bottom: 0.5rem;
            display: block;
        }
        
        .apply-model-container {
            margin-top: 1rem;
        }
        
        /* Active model badge */
        .active-model-container {
            margin: 1.5rem 0;
        }
        
        .active-model-badge {
            display: inline-flex;
            flex-direction: column;
            background: linear-gradient(135deg, #3a7bd5, #00d2ff);
            border-radius: 12px;
            padding: 0.5rem 1rem;
            color: white;
        }
        
        .active-model-title {
            font-size: 0.8rem;
            font-weight: 600;
            opacity: 0.9;
            margin-bottom: 0.25rem;
        }
        
        .active-model-details {
            display: flex;
            gap: 0.5rem;
        }
        
        .model-type-badge, .model-version-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        
        .model-type-badge {
            background: rgba(255, 255, 255, 0.2);
        }
        
        .model-version-badge {
            background: rgba(0, 0, 0, 0.1);
        }
        
        /* Prediction results */
        .prediction-container {
            margin-bottom: 2rem;
            text-align: center;
            padding: 2rem;
        }
        
        .prediction-header {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
        }
        
        .prediction-icon {
            font-size: 2rem;
            margin-right: 0.75rem;
        }
        
        .prediction-header h2 {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2c3e50;
            margin: 0;
        }
        
        .result-label {
            font-size: 3rem;
            font-weight: 700;
            margin: 1rem 0;
            color: #3a7bd5;
        }
        
        .confidence-meter {
            height: 12px;
            background-color: #e2e8f0;
            border-radius: 6px;
            overflow: hidden;
            margin: 1rem 0;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            border-radius: 6px;
            animation: fillUp 1.5s ease-out;
        }
        
        .confidence-value {
            font-size: 1rem;
            font-weight: 600;
            color: #64748b;
            margin-top: 0.5rem;
        }
        
        /* Export section */
        .export-section {
            margin-top: 2rem;
            margin-bottom: 2rem;
        }
        
        /* Welcome screen */
        .welcome-header {
            text-align: center;
            padding: 3rem 1rem;
            margin-bottom: 2rem;
            border-radius: 16px;
        }
        
        .welcome-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .welcome-subtitle {
            font-size: 1.25rem;
            color: #64748b;
            margin-top: 0.5rem;
        }
        
        /* No data container */
        .no-data-container {
            text-align: center;
            padding: 2rem;
            margin-bottom: 2rem;
        }
        
        .no-data-icon {
            font-size: 2rem;
            margin-bottom: 1rem;
        }
        
        /* Workflow steps */
        .workflow-container {
            display: flex;
            margin: 2rem 0;
        }
        
        .workflow-step {
            text-align: center;
            padding: 1.5rem 1rem;
            position: relative;
            transition: all 0.3s ease;
        }
        
        .workflow-step:hover {
            transform: translateY(-5px);
        }
        
        .step-number {
            background: linear-gradient(135deg, #3a7bd5, #00d2ff);
            color: white;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            margin: 0 auto 1rem auto;
        }
        
        .step-icon {
            font-size: 2rem;
            margin-bottom: 1rem;
        }
        
        .step-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #2c3e50;
        }
        
        .step-desc {
            font-size: 0.9rem;
            color: #64748b;
        }
        
        /* Separator */
        .animated-separator {
            height: 2px;
            background: linear-gradient(90deg, transparent, #3a7bd5, #00d2ff, #3a7bd5, transparent);
            margin: 2rem 0;
            border-radius: 1px;
            animation: gradient 6s ease infinite;
            background-size: 300% 100%;
        }
        
        /* Model cards */
        .model-cards-container {
            margin: 2rem 0;
        }
        
        .model-card {
            padding: 2rem 1.5rem;
            text-align: center;
            height: 100%;
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
        }
        
        .model-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        
        .model-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #2c3e50;
        }
        
        .model-type {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 1rem;
        }
        
        .model-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(100, 116, 139, 0.2), transparent);
            margin: 1rem 0;
            width: 80%;
            align-self: center;
        }
        
        .model-status {
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            font-weight: 600;
            margin: 1rem 0;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        
        .success-dot {
            background-color: #10b981;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
        }
        
        .warning-dot {
            background-color: #f59e0b;
            box-shadow: 0 0 10px rgba(245, 158, 11, 0.5);
        }
        
        .success {
            color: #10b981;
        }
        
        .warning {
            color: #f59e0b;
        }
        
        .model-badge {
            display: inline-flex;
            align-items: center;
            background: linear-gradient(135deg, #10b981, #06b6d4);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .badge-icon {
            background-color: rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            width: 18px;
            height: 18px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-right: 0.5rem;
            font-size: 0.7rem;
        }
        
        /* Tips section */
        .tip-container {
            padding: 1rem 0;
        }
        
        .tip-steps {
            margin-bottom: 1.5rem;
        }
        
        .tip-step {
            display: flex;
            align-items: flex-start;
            margin-bottom: 1rem;
            padding: 0.75rem;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        
        .tip-step:hover {
            background-color: rgba(58, 123, 213, 0.05);
        }
        
        .tip-step-number {
            background: linear-gradient(135deg, #3a7bd5, #00d2ff);
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.8rem;
            margin-right: 1rem;
            flex-shrink: 0;
        }
        
        .tip-step-content {
            font-size: 1rem;
        }
        
        .highlight {
            color: #3a7bd5;
            font-weight: 600;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #3a7bd5, #00d2ff);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 9999px;
            font-weight: 600;
            text-decoration: none;
            display: inline-block;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(58, 123, 213, 0.25);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(58, 123, 213, 0.3);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #3a7bd5, #00d2ff);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(58, 123, 213, 0.25);
        }
    </style>"""
