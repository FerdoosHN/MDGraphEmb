# protein_target.py
import pandas as pd

class TargetDataProcessor:
    def __init__(self, embedding_file_path, ref_file_path):
        self.embedding_file_path = embedding_file_path
        self.ref_file_path = ref_file_path
        self.data = None
        self.df_ref_new = None

    def target_load_data(self):
        # Load data from the embedding file path
        self.data = pd.read_csv(self.embedding_file_path)
        
        # Load data from the reference file path
        self.df_ref_new = pd.read_csv(self.ref_file_path, sep=r'\s+', header=None)
        self.df_ref_new.columns = ["state"]

    def merge_and_export_data(self, output_file_path):
        # Merge the two data files and export to output path
        new_data = pd.merge(self.data, self.df_ref_new, left_index=True, right_index=True)
        new_data.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    # Prompt the user to input file paths
    embedding_file_path = input("Enter the path to the embedding CSV file: ")
    ref_file_path = input("Enter the path to the reference data file: ")
    output_file_path = input("Enter the desired output file path: ")

    # Initialize the processor and run the data processing
    processor = TargetDataProcessor(embedding_file_path, ref_file_path)
    processor.target_load_data()
    processor.merge_and_export_data(output_file_path)

    print(f"Merged data has been saved to {output_file_path}")
