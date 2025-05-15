import os
import sys

def add_src_to_path():
    """
    Adds the 'src' folder to the Python path. Should be called from within a notebook
    located in the 'notebooks' folder.
    """
    notebook_dir = os.getcwd()
    src_path = os.path.abspath(os.path.join(notebook_dir, '..', 'src'))
    if src_path not in sys.path:
        sys.path.append(src_path)