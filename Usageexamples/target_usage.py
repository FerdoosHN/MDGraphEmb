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
