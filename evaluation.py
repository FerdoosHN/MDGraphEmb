import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from model_training import MultiClassClassifier

class Evaluator:
    def __init__(self):
        self.accuracy_scores = {}
        self.classification_reports = {}

    def evaluate(self, model_name, y_test, y_pred, y_score):
        """
        Evaluate model performance.
        """
        accuracy = accuracy_score(y_test, y_pred)
        self.accuracy_scores[model_name] = accuracy
        print(f'{model_name} Accuracy: {accuracy:.4f}')

        report = classification_report(y_test, y_pred, output_dict=True)
        self.classification_reports[model_name] = report

        # Calculate accuracy for each class and add it to the report
        cm = confusion_matrix(y_test, y_pred)
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(class_accuracies):
            report[str(i)]['accuracy'] = acc

        MultiClassClassifier.plot_confusion_matrix(cm, ['A', 'B', 'I', 'N'],
                                                   f'Confusion Matrix ({model_name})', accuracy)

        y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
        MultiClassClassifier.plot_roc_curve(y_test_bin, y_score, ['A', 'B', 'I', 'N'],
                                            f'ROC Curve ({model_name})', accuracy)

        MultiClassClassifier.plot_precision_recall_curve(y_test_bin, y_score, ['A', 'B', 'I', 'N'],
                                                         f'Precision-Recall Curve ({model_name})', accuracy)

    def save_results(self):
        """
        Save evaluation results.
        """
        accuracy_df = pd.DataFrame(list(self.accuracy_scores.items()), columns=['Model', 'Accuracy'])
        accuracy_df.to_csv(os.path.join("report", 'accuracy_scores.csv'), index=False)
        print("Accuracy scores saved to 'report/accuracy_scores.csv'")

        classification_reports_df = pd.concat({k: pd.DataFrame(v).transpose() for k, v in self.classification_reports.items()}, axis=0)
        classification_reports_df.to_csv(os.path.join("report", 'classification_reports.csv'))
        print("Classification reports saved to 'report/classification_reports.csv'")






