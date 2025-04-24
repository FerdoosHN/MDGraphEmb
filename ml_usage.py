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

# Set project root path and allow importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import ML pipeline module
import ml

# Define a reusable function to run the ML pipeline
def run_ml_pipeline(data_path, report_dir="report", classifiers=None):
    """
    Execute the ML classification workflow.

    Parameters:
    - data_path: str, path to the CSV file containing graph embeddings + target labels.
    - report_dir: str, directory where evaluation reports will be saved.
    - classifiers: list of str, optional list of classifiers to use.
    """
    # Ensure report output directory exists
    os.makedirs(report_dir, exist_ok=True)

    # Default list of classifiers (can be customized by user)
    if classifiers is None:
        classifiers = [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "LightGBM",
            "Neural Network",
            "CNN",
            "Support Vector"
        ]

    # Display the selected classifiers
    print("\n🔍 Running the following classifiers:")
    for clf in classifiers:
        print(f"   ✅ {clf}")

    # Run the main ML pipeline with the provided arguments
    ml.main(data_path=data_path, report_dir=report_dir, selected_classifiers=classifiers)

    # Completion message
    print("\n✅ Model training and evaluation complete.")
    print(f"📁 Reports saved in: {report_dir}")

# Allow command-line usage of the script
if __name__ == "__main__":
    import argparse

    # Setup CLI argument parser
    parser = argparse.ArgumentParser(description="Run ML classification on embedded + labeled data.")
    parser.add_argument("--data", required=True, help="Path to CSV file with embeddings and target labels")
    parser.add_argument("--report", default="report", help="Directory to save output reports")
    parser.add_argument("--classifiers", nargs='+', help="List of classifiers to run (optional)")

    # Parse CLI arguments
    args = parser.parse_args()

    # Call the main ML pipeline
    run_ml_pipeline(args.data, args.report, args.classifiers)


