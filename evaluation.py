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
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from model_training import MultiClassClassifier

class Evaluator:
    def __init__(self):
        self.accuracy_scores = {}
        self.classification_reports = {}

    def evaluate(self, model_name, y_test, y_pred, y_score, class_labels):
        """
        Evaluate model performance.
        """
        try:
            accuracy = accuracy_score(y_test, y_pred)
            print(f'{model_name} Accuracy: {accuracy:.4f}')
            self.accuracy_scores[model_name] = accuracy

            report = classification_report(y_test, y_pred, output_dict=True)
            self.classification_reports[model_name] = report  # Save early to avoid total loss on error

            # Calculate accuracy for each class and add it to the report only if the class exists
            cm = confusion_matrix(y_test, y_pred)
            class_accuracies = cm.diagonal() / cm.sum(axis=1)
            for i, acc in enumerate(class_accuracies):
                class_str = str(i)
                if class_str in report:
                    report[class_str]['accuracy'] = acc

            # Use actual class label names dynamically
            MultiClassClassifier.plot_confusion_matrix(cm, class_labels,
                                                       f'Confusion Matrix ({model_name})', accuracy)

            # Use label_binarize with dynamic class index range
            y_test_bin = label_binarize(y_test, classes=range(len(class_labels)))

            MultiClassClassifier.plot_roc_curve(y_test_bin, y_score, class_labels,
                                                f'ROC Curve ({model_name})', accuracy)

            MultiClassClassifier.plot_precision_recall_curve(y_test_bin, y_score, class_labels,
                                                             f'Precision-Recall Curve ({model_name})', accuracy)

        except Exception as e:
            print(f"❌ Evaluation failed for {model_name}: {e}")

    def save_results(self):
        """
        Save evaluation results.
        """
        accuracy_df = pd.DataFrame(list(self.accuracy_scores.items()), columns=['Model', 'Accuracy'])
        accuracy_df.to_csv(os.path.join("report", 'accuracy_scores.csv'), index=False)
        print("✅ Accuracy scores saved to 'report/accuracy_scores.csv'")

        if self.classification_reports:
            classification_reports_df = pd.concat(
                {k: pd.DataFrame(v).transpose() for k, v in self.classification_reports.items()},
                axis=0
            )
            classification_reports_df.to_csv(os.path.join("report", 'classification_reports.csv'))
            print("✅ Classification reports saved to 'report/classification_reports.csv'")
        else:
            print("⚠️ No classification reports to save. Possibly all classifiers failed.")







