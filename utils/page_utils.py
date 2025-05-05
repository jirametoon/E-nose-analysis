"""
Main page utilities for the E-nose Analytics application.
This module provides functions for rendering different pages in the application.
"""
from .pages import home_page, info_page, process_page, result_page

def page_home():
    """
    Render the home page with feature overview and getting started section.
    """
    home_page.render()

def page_info():
    """
    Render the information page with explanations about E-nose technology.
    """
    info_page.render()

def page_process():
    """
    Render the process page for data uploading and preprocessing.
    """
    process_page.render()

def page_result():
    """
    Render the results page with model predictions and visualizations.
    """
    result_page.render()