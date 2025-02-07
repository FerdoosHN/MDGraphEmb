
# **MDGraphEmb: A Toolkit for Encoding Molecular Dynamics Ensembles with Graph Embedding**

## **Motivation**  
Molecular Dynamics (MD) simulations are essential for investigating protein dynamics and function.
Although there have been significant advances in integrating simulation techniques and machine learning, there are
still challenges in selecting the best data representation for learning. Graph embedding is a powerful computational
method that automatically learns low-dimensional representations of nodes in a graph while preserving graph topology
and node properties, thereby bridging graph structures and machine learning methods. Graph embeddings have great
potential for efficient representation of MD simulation data and studying protein dynamics.

---

## **Overview**  
We introduce MDGraphEmb, a Python library built on MDAnalysis, designed to facilitate the conversion of
protein conformation data into graph-based representations and graph embeddings. This enables the compression of large
noisy simulation trajectories into tabular formats suitable for machine learning. MDGraphEmb provides a framework
that supports a range of graph embedding techniques and machine learning models to create workflows to analyse protein
dynamics and identify important protein conformations. Graph embedding effectively captures and compresses structural
information from MD simulation data, making it applicable to diverse downstream machine-learning classification tasks.
An application is presented for encoding and detecting important protein conformations from molecular dynamics
simulations to classify functional states. Furthermore, a performance comparison of different graph embedding methods
in combination with machine learning models is reported.

---

## **Installation**

1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/FerdoosHN/MDGraphEMB.git  
   cd MDGraphEMB  
   ```

2. **Create a Virtual Environment and Activate It:**  
   ```bash
   python -m venv mdgraphenv  
   source mdgraphenv/bin/activate  # On Windows: mdgraphenv\Scripts\activate  
   ```

3. **Install the Required Dependencies:**  
   ```bash
   pip install -r requirements.txt  
   ```

---

## **Prerequisites**  
Ensure you have **Python 3.12.4** or later installed. You can check your Python version using:  
```bash
python --version  
```

---

## **Required Packages**  
The following packages are necessary for using **MDGraphEmb**:

```
numpy  
pandas  
scikit-learn  
torch  
torch-geometric  
MDAnalysis  
networkx  
matplotlib  
seaborn  
joblib  
imblearn  
xgboost  
lightgbm  
tensorflow  
node2vec  
```

To install them manually:  
```bash
pip install -r requirements.txt  
```

---

## **Key Features**  

- **Graph Embedding Methods:**  
  - **Node2Vec**: Generates random walks on graphs to learn embeddings.  
  - **GraphSAGE**: Samples and aggregates neighborhood information to learn node representations.  
  - **GCN** (Graph Convolutional Networks): Captures graph structure using convolutional layers.  
  - **GAT** (Graph Attention Networks): Utilizes attention mechanisms for node aggregation.  

- **Machine Learning Models Supported:**  
  - **Logistic Regression**: A simple baseline classifier.  
  - **Random Forest**: A powerful ensemble model for classification.  
  - **Support Vector Machines (SVM)**: For binary and multi-class classification tasks.  
  - **XGBoost and LightGBM**: Gradient boosting models for classification.  
  - **Neural Networks (NNs) and CNNs**: For deep learning-based classification.  

- **Preprocessing Utilities:**  
  - **SMOTE**: Oversamples imbalanced datasets.  
  - **MDAnalysis loaders**: Efficient loading of protein conformational data.  
  - **Data Standardization**: Ensures normalized input for ML models.  

---

## **Usage Workflow**

### **Input**
- **Trajectory file**: The Molecular Dynamics simulation trajectory (e.g., `.xtc`).
- **Topology file**: The corresponding topology file (e.g., `.gro`).
- **Target data file**:  
  File specifying target values, such as:
  - `target.dat`

- **Atom selection** *(optional)*: Default is `"name CA"` (alpha-carbon atoms).

---

### **Output**
The following output files will be generated during the embedding and machine learning workflow:

#### **1. Protein Embeddings**
- **File**: `embedding.csv`  
  Contains the embedded representation of the protein for each frame in the trajectory, depending on the embedding method and frame size.

#### **2. Target Embeddings**
- **File**: `target_embedded_data.csv`  
  Contains the target embeddings used for the machine learning classification task, depending on the embedding method and frame size.

#### **3. Classification Results**
- **File**: `accuracy_scores.csv`  
  Includes accuracy scores for different machine learning models.

#### **4. Classification Report**
- **File**: `classification_report.csv`  
  Contains detailed metrics, including accuracy, precision, recall, F1-score, and support for each class.

#### **5. Confusion Matrix**
- **File**: `confusion_matrix.png`  
  Visual representation of the confusion matrix.

#### **6. Precision-Recall Curve**
- **File**: `precision_recall_curve.png`  
  Visual representation of the trade-off between precision and recall.

#### **7. Train/Test Data and Predictions**
- **File**: `data_with_predictions.csv`  
  Contains data with predictions generated by all machine learning models, including the exact frames used for training and testing.

#### **8. ROC Curve**
- **File**: `roc_curve.png`  
  Shows the sensitivity and specificity of different models using the ROC curve.

---

### **File Structure Example**
Here’s an example of what the input and output files might look like in your project directory:

```plaintext
MDGraphEMB/
├── input/
│   ├── trajectory.xtc
│   ├── topology.gro
│   └── target.dat
├── output/
│   ├── embedding.csv
│   ├── target_embedded_data.csv
│   ├── accuracy_scores.csv
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   ├── data_with_predictions.csv
│   ├── precision_recall_curve.png
│   └── roc_curve.png
```

---



## **Authors**  

- Ferdoos Hossein Nezhad  
- Namir Oues  
- Massimiliano Meli  
- Alessandro Pandini  

---

## **License**  
This project is licensed under the **GPL-3.0 License**.
