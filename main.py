"""
GMMA (Guppy Multiple Moving Average) Fund Analysis Application

This application displays GMMA charts for Chinese funds using data from akshare.
It supports analyzing individual funds or automatically scanning for funds with recent signals.
"""
import streamlit as st
from ui.components import setup_page, setup_sidebar
from ui.scan_mode import run_scan_mode
from ui.individual_mode import run_individual_mode

def main():
    """
    Main application entry point.
    """
    # Setup page configuration and title
    setup_page()
    
    # Setup sidebar and get settings
    settings = setup_sidebar()
    
    # Run appropriate mode based on settings
    if settings['analysis_mode'] == "基金全扫描":
        run_scan_mode(settings)
    else:  # "指定基金分析"
        run_individual_mode(settings)

if __name__ == "__main__":
    main() 