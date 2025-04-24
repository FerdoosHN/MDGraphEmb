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


# Minimal working example of the entire pipeline 
"""
Tutorial: Using MDGraphEMB to Generate GraphSAGE Embeddings and Classify Protein States

This tutorial walks through the entire pipeline:
1. Generate graph embeddings from protein MD trajectories using GraphSAGE (recommended embedding method)
2. Merge those embeddings with target conformational state labels
3. Train machine learning classifiers to predict conformational states

Expected directory structure:
- ADKData/               → Contains .xtc (trajectory) and .gro (topology)
- embedded_data/         → Stores output from GraphSAGE
- target_embedded_data/  → Stores merged embeddings + target labels
- report/                → Final evaluation results and plots

Author: Ferdoos Hossein Nezhad
"""

import os
import sys

# -------------------------------------
# Setup: Add project root to Python path
# -------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(project_root)

# ===============
# STEP 1: Embeddings
# ===============
print("🔷 Step 1: Generating GraphSAGE embeddings...")

import protein_emb

# Define input files (trajectory + topology)
trajectory_file = os.path.join(project_root, 'ADKData', 'R02_4AKE_DM_prod_T300_protein_dt200.xtc')
topology_file = os.path.join(project_root, 'ADKData', 'R02_4AKE_DM_prod_T300_protein.gro')

# Output path for generated embeddings
embedding_output_file = os.path.join(project_root, 'embedded_data', 'graphsage_embeddings.csv')

# Create the directory if needed
os.makedirs(os.path.dirname(embedding_output_file), exist_ok=True)

# Run GraphSAGE embedding pipeline
protein_emb.run_graphsage(trajectory_file, topology_file, embedding_output_file)
print("✅ GraphSAGE embeddings generated and saved.")


# ===============================
# STEP 2: Merge with Target Labels
# ===============================
print("\n🔷 Step 2: Merging embeddings with target labels...")

from protein_target import TargetDataProcessor

# Define path to target label data (frame-wise states)
ref_file_path = os.path.join(project_root, 'ADKData', 'R02_4AKE_DM_prod_T300_dt200_target_value.dat')
labeled_output_file = os.path.join(project_root, 'target_embedded_data', 'graphsage5000.csv')

# Create the output directory if not exists
os.makedirs(os.path.dirname(labeled_output_file), exist_ok=True)

# Load and merge embedding + target
processor = TargetDataProcessor(embedding_output_file, ref_file_path)
processor.target_load_data()
processor.merge_and_export_data(labeled_output_file)
print("✅ Embeddings successfully merged with target labels.")


# ===================================
# STEP 3: ML Classification + Evaluation
# ===================================
print("\n🔷 Step 3: Running machine learning classification...")

import ml

# Directory for output reports (accuracy, plots, etc.)
report_dir = os.path.join(project_root, 'report')
os.makedirs(report_dir, exist_ok=True)

# Choose classifiers to include in the pipeline
selected_classifiers = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM"
]

# Run model training and evaluation
ml.main(
    data_path=labeled_output_file,
    report_dir=report_dir,
    selected_classifiers=selected_classifiers
)

print("\n✅ ML classification and evaluation complete.")
print("📊 All reports and plots are saved in the 'report/' folder.")
