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

# Determine the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)  # Add project root to sys.path for imports

from protein_target import TargetDataProcessor

# Define file paths relative to the project root
embedding_file_path = os.path.join(project_root, 'embedded_data', 'graphsage_embeddings.csv')
ref_file_path = os.path.join(project_root, 'ADKData', 'R02_4AKE_DM_prod_T300_dt200_target_value.dat')
output_file_path = os.path.join(project_root, 'target_embedded_data', 'graphsage5000.csv')

# Initialize the processor and run the data processing
processor = TargetDataProcessor(embedding_file_path, ref_file_path)
processor.target_load_data()
processor.merge_and_export_data(output_file_path)

print(f"Merged data has been saved to {output_file_path}")
