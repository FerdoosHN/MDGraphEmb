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

# Set project root and make internal modules importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import GraphSAGE embedding runner from the MDGraphEMB pipeline
import protein_emb

def generate_graphsage_embeddings(trajectory_file, topology_file, output_file):
    """
    Run GraphSAGE embedding generation from protein MD trajectory.

    Parameters:
    - trajectory_file: str, path to the input trajectory (.xtc)
    - topology_file: str, path to the topology (.gro)
    - output_file: str, destination path for the CSV output
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Call the main embedding generation function
    protein_emb.run_graphsage(trajectory_file, topology_file, output_file)

    # Confirmation message
    print(f"✅ Embeddings saved to {output_file}")

# Enable command-line usage
if __name__ == "__main__":
    import argparse

    # Setup command-line arguments
    parser = argparse.ArgumentParser(description="Generate GraphSAGE embeddings.")
    parser.add_argument("--trajectory", required=True, help="Path to trajectory (.xtc) file")
    parser.add_argument("--topology", required=True, help="Path to topology (.gro) file")
    parser.add_argument("--output", required=True, help="Output CSV path for embeddings")

    # Parse arguments
    args = parser.parse_args()

    # Call the embedding generator
    generate_graphsage_embeddings(args.trajectory, args.topology, args.output)

