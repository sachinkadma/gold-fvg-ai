"""
Streamlit Cloud Entry Point Bridge
Redirects root-level execution directly to app/streamlit_app.py
ensuring 100% compatibility with Streamlit Cloud default configurations.
"""
import os
import sys
import runpy

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

target_script = os.path.join(REPO_ROOT, "app", "streamlit_app.py")
runpy.run_path(target_script, run_name="__main__")
