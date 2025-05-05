"""
E-Nose Analytics Application

This is the main entry point for the E-Nose Analytics application,
which provides tools for electronic nose data visualization and analysis.
"""

import streamlit as st
from streamlit_option_menu import option_menu
from utils.page_utils import page_home, page_info, page_process, page_result

# Constants
APP_TITLE = "E-NOSE"
LOGO_PATH = "assets/logo_nose.png"
HEADER_TEXT = "👃 ELECTRONIC-NOSE"

# Menu configuration
MENU_ITEMS = ["Home", "Info", "Process", "Result"]
MENU_ICONS = ["house", "info-circle", "gear", "check2-circle"]

# CSS for global header animations and styling
HEADER_CSS = """
    /* Animation keyframes */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }

    /* Global header style */
    .global-header {
        font-family: 'Limelight', cursive;
        font-size: 3.5rem;
        text-align: center;
        margin: 1rem auto;
        background: linear-gradient(90deg, #ff7b00, #ff9a44, #ffb347, #ff9a44, #ff7b00);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 5px rgba(255,123,0,0.2);
        animation: gradient 6s ease infinite, pulse 3s ease-in-out infinite;
        display: block;
        width: 100%;
    }

    /* Underline accent with animation */
    .global-header::after {
        content: "";
        display: block;
        width: 180px;
        height: 4px;
        background: linear-gradient(90deg, #ff7b00, #ff9a44, #ffb347, #ff9a44, #ff7b00);
        background-size: 300% 100%;
        margin: 0.5rem auto 1.5rem auto;
        border-radius: 4px;
        animation: gradient 6s ease infinite;
        box-shadow: 0 2px 8px rgba(255,123,0,0.3);
    }
"""


def setup_page():
    """Configure initial Streamlit page settings."""
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    st.markdown(f"<style>{HEADER_CSS}</style>", unsafe_allow_html=True)


def render_header():
    """Display the animated global header."""
    st.markdown(
        f'<h1 class="global-header">{HEADER_TEXT}</h1>',
        unsafe_allow_html=True
    )


def render_sidebar():
    """Setup and render the sidebar navigation menu."""
    # Display logo
    st.sidebar.image(LOGO_PATH, use_container_width=True)
    
    # Create navigation menu
    with st.sidebar:
        selected_page = option_menu(
            menu_title=None,
            options=MENU_ITEMS,
            icons=MENU_ICONS,
            menu_icon="cast",
            default_index=0,
            orientation="vertical",
        )
    
    return selected_page


def route_to_page(page):
    """Route to the selected page based on menu selection.
    
    Args:
        page (str): The selected page name from the sidebar menu
    """
    # Map pages to their corresponding functions
    page_routes = {
        "Home": page_home,
        "Info": page_info,
        "Process": page_process,
        "Result": page_result
    }
    
    # Call the appropriate page function
    if page in page_routes:
        page_routes[page]()
    else:
        st.error(f"Page '{page}' not found.")


def main():
    """Main application entry point."""
    # Setup page configuration
    setup_page()
    
    # Render header
    render_header()
    
    # Setup sidebar and get selected page
    selected_page = render_sidebar()
    
    # Route to the selected page
    route_to_page(selected_page)


if __name__ == "__main__":
    main()
