from __future__ import annotations

import streamlit as st


def main():
    """The central entry point for the Bing Experimentation Suite dashboard.

    This multi page application provides a comprehensive interface for
    online controlled experimentation diagnosis, metric health monitoring,
    and variance reduction benchmarking. It is designed for both
    statisticians and product owners to validate experimental integrity.
    """
    st.set_page_config(
        page_title="Bing Experimentation Suite",
        page_icon="🔍",
        layout="wide"
    )

    st.title("Bing Experimentation Suite")
    st.markdown("""
    Welcome to the Bing Experimentation Suite. This platform provides a production ready 
    interface for analyzing online experiments and monitoring metric health. 
    We implement advanced statistical methods such as CUPED and novelty effect detection 
    to ensure precise and reliable product decisions.
    """)

    st.sidebar.title("Navigation")
    st.sidebar.info("Select a page from the sidebar to view detailed analyses.")

    # Streamlit automatically handles multi page navigation through the 'pages/' folder.
    # The sidebar will show the files in the 'pages/' directory by default.


if __name__ == "__main__":
    main()
