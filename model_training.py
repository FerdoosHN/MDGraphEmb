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
import joblib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

class MultiClassClassifier:
    def __init__(self, model, filename, label_encoder):
        self.model = model
        self.filename = os.path.join("report", filename)
        self.label_encoder = label_encoder

    def fit(self, x_train, y_train, sample_weight=None):
        # Fit the model with or without sample_weight
        if sample_weight is not None:
            self.model.fit(x_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def save_model(self):
        joblib.dump(self.model, self.filename)
        print(f"Model saved as: {self.filename}")

    def predict_labels(self, x):
        numeric_predictions = self.predict(x)
        return self.label_encoder.inverse_transform(numeric_predictions)

    @staticmethod
    def plot_confusion_matrix(cm, class_labels, title, accuracy):
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(cm.shape[0], cm.shape[1])
        sns.heatmap(cm / np.sum(cm), annot=labels, fmt='', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f"{title}\nAccuracy: {accuracy:.4f}")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join("report", f'{title.replace(" ", "_").lower()}_confusion_matrix.png'))
        plt.close() 
       

    @staticmethod
    def plot_roc_curve(y_test, y_score, class_labels, title, accuracy):
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        n_classes = y_test.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                      ''.format(class_labels[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f"{title}\nAccuracy: {accuracy:.4f}")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join("report", f'{title.replace(" ", "_").lower()}_roc_curve.png'))
        plt.close() 
      

    @staticmethod
    def plot_precision_recall_curve(y_test, y_score, class_labels, title, accuracy):
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        n_classes = y_test.shape[1]
        precision = dict()
        recall = dict()

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
            plt.plot(recall[i], precision[i], lw=2,
                     label='Precision-Recall curve of class {0}'.format(class_labels[i]))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f"{title}\nAccuracy: {accuracy:.4f}")
        plt.legend(loc="best")
        plt.savefig(os.path.join("report", f'{title.replace(" ", "_").lower()}_precision_recall_curve.png'))
        plt.close() 
    
