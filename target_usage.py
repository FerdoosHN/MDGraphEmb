"""
@release_date  : $release_date
@version       : $release_version
@author        : Ferdoos Hossein Nezhad

This file is part of the MDGraphEMB software 
(https://github.com/FerdoosHN/MDGraphEMB).
Copyright (c) 2025 Ferdoos Hossein Nezhad and Alessandro Pandini.

This program is free software: you can redistribute it and/or modify 
it under the terms of the GNU General Public License as published by  
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
General Public License for more details.

You should have received a copy of the GNU General Public License 
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import sys

# Set the project root directory so internal modules can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the target label processor from the pipeline
from protein_target import TargetDataProcessor

def merge_targets(embedding_file, target_file, output_file):
    """
    Merge protein embeddings with their conformational target labels.

    Parameters:
    - embedding_file: str, path to the CSV file with graph embeddings
    - target_file: str, path to the target label file (.dat format)
    - output_file: str, destination CSV path to save merged data
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Initialize and run the data processor
    processor = TargetDataProcessor(embedding_file, target_file)
    processor.target_load_data()
    processor.merge_and_export_data(output_file)

    # Confirm completion
    print(f"✅ Merged data saved to {output_file}")

# Enable script to be used from command line
if __name__ == "__main__":
    import argparse

    # Define CLI arguments
    parser = argparse.ArgumentParser(description="Merge embeddings with target labels.")
    parser.add_argument("--embedding", required=True, help="Path to the embeddings CSV file")
    parser.add_argument("--target", required=True, help="Path to the target .dat file")
    parser.add_argument("--output", required=True, help="Output CSV path for merged data")

    # Parse CLI inputs and call the function
    args = parser.parse_args()
    merge_targets(args.embedding, args.target, args.output)
