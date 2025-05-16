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
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from ml_data_preprocessing import load_and_prepare_data, split_data
from model_training import MultiClassClassifier
from evaluation import Evaluator

def get_class_weights(y):
    class_support = {label: np.sum(y == label) for label in np.unique(y)}
    total_samples = len(y)
    class_weights = {cls: total_samples / (len(class_support) * count) for cls, count in class_support.items()}
    return class_weights

def build_nn_model(input_shape, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main(data_path, report_dir="report", selected_classifiers=None):
    df, x, y_encoded, label_encoder = load_and_prepare_data(data_path)
    x_train, y_train, x_test, y_test, test_indices = split_data(x, y_encoded)

    smote = SMOTE(random_state=42, k_neighbors=1)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    class_weights = get_class_weights(y_train_resampled)

    classifiers = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "Random Forest": RandomForestClassifier(class_weight='balanced'),
        "XGBoost": XGBClassifier(),
        "LightGBM": LGBMClassifier(class_weight='balanced'),
        "Neural Network": 'neural_network',
        "CNN": 'cnn',
        "Support Vector": SVC(probability=True, class_weight='balanced')
    }

    if selected_classifiers:
        classifiers = {name: classifiers[name] for name in selected_classifiers if name in classifiers}

    evaluator = Evaluator()
    df['set_type'] = 'train'
    df.loc[test_indices, 'set_type'] = 'test'

    for name, model in classifiers.items():
        print(f"\nRunning {name}...")
        filename = f'model_{name.lower().replace(" ", "_")}.job'

        if name in ["Neural Network", "CNN", "Support Vector", "Logistic Regression"]:
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train_resampled)
            x_test_scaled = scaler.transform(x_test)
        else:
            x_train_scaled = x_train_resampled
            x_test_scaled = x_test

        x_train_resampled_reshaped = np.expand_dims(x_train_scaled, axis=-1)
        x_test_reshaped = np.expand_dims(x_test_scaled, axis=-1)

        if name == "Neural Network":
            nn_model = build_nn_model(x_train_scaled.shape[1], len(np.unique(y_encoded)))
            y_train_resampled_cat = to_categorical(y_train_resampled, len(np.unique(y_encoded)))

            nn_model.fit(x_train_scaled, y_train_resampled_cat,
                         epochs=30, batch_size=32, validation_split=0.2,
                         class_weight=class_weights,
                         callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

            nn_model.save(os.path.join(report_dir, 'model_neural_network.h5'))

            predictions = np.argmax(nn_model.predict(x_test_scaled), axis=-1)
            y_score = nn_model.predict(x_test_scaled)

            df.loc[test_indices, f'{name}_pred'] = label_encoder.inverse_transform(predictions)
            evaluator.evaluate(name, y_test, predictions, y_score, class_labels=label_encoder.classes_)

        elif name == "CNN":
            cnn_model = build_cnn_model(x_train_resampled_reshaped.shape[1], len(np.unique(y_encoded)))
            y_train_resampled_cat = to_categorical(y_train_resampled, len(np.unique(y_encoded)))

            cnn_model.fit(x_train_resampled_reshaped, y_train_resampled_cat,
                          epochs=30, batch_size=32, validation_split=0.2,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

            cnn_model.save(os.path.join(report_dir, 'model_cnn.h5'))

            predictions = np.argmax(cnn_model.predict(x_test_reshaped), axis=-1)
            y_score = cnn_model.predict(x_test_reshaped)

            df.loc[test_indices, f'{name}_pred'] = label_encoder.inverse_transform(predictions)
            evaluator.evaluate(name, y_test, predictions, y_score, class_labels=label_encoder.classes_)

        else:
            multi_classifier = MultiClassClassifier(model, filename, label_encoder)
            multi_classifier.fit(x_train_scaled, y_train_resampled)

            df.loc[test_indices, f'{name}_pred'] = multi_classifier.predict_labels(x_test_scaled)

            y_pred = multi_classifier.predict(x_test_scaled)
            y_score = multi_classifier.predict_proba(x_test_scaled)

            evaluator.evaluate(name, y_test, y_pred, y_score, class_labels=label_encoder.classes_)
            multi_classifier.save_model()

        df_with_pred = df[['set_type', 'state', f'{name}_pred']]
        df_with_pred.to_csv(os.path.join(report_dir, f'data_with_predictions_{name.lower().replace(" ", "_")}.csv'), index=False)
        print(f"Dataset with {name} predictions saved to 'report/data_with_predictions_{name.lower().replace(' ', '_')}.csv'")

    evaluator.save_results()
