"""
This file contains the paths for the project and some helper functions.
"""

import os
import json
from glob import glob


def return_paths():
    """
    Return paths for the project.
    """
    # Load configuration from config.json
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    docs_path = config['docs_path']
    file_list = glob(os.path.join(docs_path, "*"))
    vector_persist_dir = config['vector_persist_dir']
    docs_output_dir = config['docs_output_dir']
    docs_output_path = os.path.join(docs_output_dir, 'docs.pkl')
    return file_list, docs_output_path, docs_output_dir, vector_persist_dir


# Run this to generate the paths
(
    file_list, 
    docs_output_path, 
    docs_output_dir,
    vector_persist_dir,
    ) = return_paths()

if not os.path.exists(vector_persist_dir):
    os.makedirs(vector_persist_dir)

if not os.path.exists(docs_output_dir):
    os.makedirs(docs_output_dir)
