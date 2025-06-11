# **MDGraphEmb: A Toolkit for Graph Embedding and Classification of Protein Conformational Ensembles**

## **Motivation**  
Molecular Dynamics (MD) simulations are essential for investigating protein dynamics and function. Although significant advances have been made in integrating simulation techniques and machine learning, there are still challenges in selecting the most suitable data representation for learning. Graph embedding is a powerful computational method that automatically learns low-dimensional representations of nodes in a graph while preserving graph topology and node properties, thereby bridging graph structures and machine learning methods. Graph embeddings hold great potential for efficiently representing MD simulation data and studying protein dynamics.

---

## **Overview**  
We introduce MDGraphEmb, a Python library built on MDAnalysis, designed to facilitate the conversion of protein conformation data into graph-based representations and graph embeddings. This enables the compression of large noisy simulation trajectories into tabular formats suitable for machine learning. MDGraphEmb provides a framework that supports a range of graph embedding techniques and machine learning models, enabling the creation of workflows to analyse protein dynamics and identify important protein conformations. Graph embedding effectively captures and compresses structural information from protein MD simulation data, making it applicable to diverse downstream machine-learning classification tasks. An application is presented for encoding and detecting important protein conformations from molecular dynamics simulations to classify functional states. Furthermore, a performance comparison of different graph embedding methods in combination with machine learning models is reported.

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

## Data availability
Relevant data underpinning this publication can be accessed from Brunel University London’s data repository under CC BY licence:
https://doi.org/10.17633/rd.brunel.c.7664645

---

## **Key Features**  

- **Graph Embedding Methods:**  
  - **Node2Vec**: Generates random walks on graphs to learn embeddings.  
  - **GraphSAGE** (recommended): Samples and aggregates neighbourhood information to learn node representations.  
  - **GCN** (Graph Convolutional Networks): Captures graph structure using convolutional layers.  
  - **GAT** (Graph Attention Networks): Utilises attention mechanisms for node aggregation.  

- **Machine Learning Models Supported:**  
  - **Logistic Regression**: A simple baseline classifier.  
  - **Random Forest**: A powerful ensemble model for classification.  
  - **Support Vector Machines (SVM)**: For binary and multi-class classification tasks.  
  - **XGBoost and LightGBM**: Gradient boosting models for classification.  
  - **Neural Networks (NNs) and CNNs**: For deep learning-based classification.  

>  If you are working with a large dataset or need faster performance, we recommend starting with:
> 
> - Logistic Regression
> - Random Forest
> - XGBoost
> - LightGBM
>
> If you are not constrained by runtime and want to explore deeper representations, you may also include:
> 
> - Neural Network
> - CNN
> - Support Vector Machine (SVM)

- **Preprocessing Utilities:**  
  - **SMOTE**: Oversamples imbalanced datasets.  
  - **MDAnalysis loaders**: Efficient loading of protein conformational data.  
  - **Data Standardisation**: Ensures normalised input for ML models.  

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
├── ADKData/
│   ├── R02_4AKE_DM_prod_T300_protein_dt200.xtc
│   ├── R02_4AKE_DM_prod_T300_protein.gro
│   └── R02_4AKE_DM_prod_T300_dt200_target_value.dat
├── embedded_data/
│   └── graphsage_embeddings.csv
├── target_embedded_data/
│   └── graphsage5000.csv
├── report/
│   ├── accuracy_scores.csv
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   ├── precision_recall_curve.png
│   ├── roc_curve.png
│   └── data_with_predictions.csv
```

## **Workflow Overview**

The table below outlines the main steps of the MDGraphEMB pipeline and their corresponding script names:

| Pipeline Step                   | Script Name         | Description                                                                          |
|--------------------------------|---------------------|--------------------------------------------------------------------------------------|
| Step 1: Embedding Generation   | embedding_usage.py  | Generates GraphSAGE embeddings from MD trajectory data.                             |
| Step 2: Merge with Targets     | target_usage.py     | Merges the generated embeddings with target conformational state labels.            |
| Step 3: ML Training & Evaluation | ml_usage.py       | Trains machine learning classifiers and evaluates their performance.                |
| Full Tutorial Example          | tutorialexample.py  | A complete walkthrough that runs all steps in sequence.                             |

**Explanation:**
- This table summarises the typical MDGraphEMB pipeline that users may follow for classifying protein conformational states.
- You may customise each step by modifying the corresponding usage script.
- You can also include other embedding methods such as Node2Vec, GCN, or GAT, or additional classifiers such as `Neural Network`, `CNN`, or `SVM` that are provided in the toolkit.
- Several `.sh` bash files (e.g., `run_embedding.sh`, `run_target.sh`, `run_ml.sh`) are provided to run and automate the workflow steps efficiently. You can easily modify them.
- A simple tutorial script (`tutorialexample.py`) is provided to demonstrate the entire pipeline.
- Note: If users only wish to generate graph embeddings from an MD dataset, the target.dat file is not required. This file is only necessary when performing supervised learning tasks that involve labelled target data.

---
## **Performance Benchmarks**

The following table summarises the runtime and memory usage for key steps in the MDGraphEmb pipeline across different frame counts for the ADK system. All measurements were obtained on a **MacBook Pro (14", Nov 2023)** with an **Apple M3 Pro chip** and **18 GB RAM**, using the **GraphSAGE embedding method** and selected machine learning models (**Logistic Regression**, **Random Forest**, **LightGBM**, **XGBoost**):

| **Step**                         | **Frames** | **Time (s)** *(min)*     | **Memory (MB)** |
|----------------------------------|------------|---------------------------|-----------------|
| **GraphSAGE Embedding**          | 5000       | 144.60 *(~2.4 min)*       | 3831.91         |
|                                  | 10000      | 298.86 *(~5.0 min)*       | 1169.09         |
|                                  | 25000      | 802.49 *(~13.4 min)*      | 1844.83         |
|                                  | 50000      | 1650.00 *(~27.5 min)*     | 3200.00         |
| **Merging Targets**              | 5000       | 34.57 *(~0.6 min)*        | 3937.53         |
|                                  | 10000      | 66.07 *(~1.1 min)*        | 1682.58         |
|                                  | 25000      | 177.12 *(~3.0 min)*       | 2262.70         |
|                                  | 50000      | 360.00 *(~6.0 min)*       | 2500.00         |
| **ML Training + Evaluation**     | 5000       | 379.81 *(~6.3 min)*       | 3107.89         |
| *(LR, RF, LGBM, XGB)*            | 10000      | 788.12 *(~13.1 min)*      | 2021.38         |
|                                  | 25000      | 2660.46 *(~44.3 min)*     | 1739.50         |
|                                  | 50000      | 5400.00 *(~90.0 min)*     | 1600.00         |

> **Note:**  Memory usage may vary depending on system activity and background processes. Users with a standard modern laptop (8–16 GB RAM) should experience smooth performance up to **25,000 frames**. 


## **Authors**  

- Ferdoos Hossein Nezhad  
- Namir Oues  
- Massimiliano Meli  
- Alessandro Pandini  

---

## **License**  
This project is licensed under the **GPL-3.0 License**.
