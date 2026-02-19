"""
Shared sidebar — call render_sidebar() from every page to get consistent
CSS and credit footer across the entire dashboard.
"""

import streamlit as st
from pathlib import Path


def render_sidebar():
    """Load custom CSS. Call this at the top of every page."""

    css_path = Path(__file__).resolve().parent.parent / "assets" / "style.css"
    logo_path = "assets/eeg.png"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )
    
    st.logo(logo_path, size="large")


def render_sidebar_footer():
    """Render the credit footer at the bottom of the sidebar."""
    st.sidebar.markdown(
        """
        <div style="text-align:center; padding: 0.3rem 0;">
            <div style="font-size: 0.72rem; color: #484f58; margin-bottom: 6px;">
                Developed by
            </div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #c9d1d9;">
                Amir Rafe
            </div>
            <a href="mailto:amir.rafe@txstate.edu"
               style="font-size: 0.72rem; color: #58a6ff; text-decoration: none;">
                amir.rafe@txstate.edu
            </a>
            <div style="margin-top: 6px;">
                <span style="font-size: 0.65rem; color: #30363d;">━━━</span>
            </div>
            <a href="https://pozapas.github.io/" target="_blank"
               style="font-size: 0.7rem; color: #58a6ff; text-decoration: none; margin-top: 4px; display:inline-block;">
                🌐 pozapas.github.io
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
