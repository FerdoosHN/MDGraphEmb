import os
import sys

# Set the main project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import ml

# Define paths
data_path = '/Users/ferdooshosseinnezhad/Projects/MDGraphEMB/target_embedded_data/graphsage5000.csv'
report_dir = 'report'

# Ensure the report directory exists
os.makedirs(report_dir, exist_ok=True)

# Available classifiers in `ml.py`:
# classifiers = {
#     "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
#     "Random Forest": RandomForestClassifier(class_weight='balanced'),
#     "XGBoost": XGBClassifier(),
#     "LightGBM": LGBMClassifier(class_weight='balanced'),
#     "Neural Network": 'neural_network',
#     "CNN": 'cnn',
#     "Support Vector": SVC(probability=True, class_weight='balanced')
# }

# Choose classifiers to run from the list above
# These are the classifiers I have chosen:
selected_classifiers = ["Logistic Regression", "XGBoost", "LightGBM", "Random Forest", "Neural Network", "CNN", "Support Vector"]

# Display selected classifiers
print("\nRunning the following classifiers:")
for model_name in selected_classifiers:
    print(f"- {model_name}")

# Run the main function with only the selected classifiers
ml.main(data_path, report_dir, selected_classifiers=selected_classifiers)
print(f"\nModel training and evaluation completed. Reports saved in '{report_dir}'.")



