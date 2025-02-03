
# **MDGraphEmb: A Toolkit for Encoding Molecular Dynamics Ensembles with Graph Embedding**

## **Motivation**  
Molecular Dynamics (MD) simulations are essential for investigating protein dynamics and function. Although there have been significant advances in the integration of simulation techniques and machine learning, there are still challenges in selecting the best data representation for learning.  

Graph embedding is a powerful computational method that automatically learns low-dimensional representations of nodes in a graph while preserving graph topology and node properties, thereby bridging graph structures and machine learning methods. Graph embeddings have great potential for efficient representation of MD simulation data and the study of protein dynamics.

---

## **Overview**  
We introduce **MDGraphEmb**, a Python library built on **MDAnalysis**, designed to facilitate the conversion of protein conformation data into graph-based representations and graph embeddings. This enables the compression of large noisy simulation trajectories into tabular formats suitable for machine learning.  

**MDGraphEmb** provides a framework that supports a range of graph embedding techniques and machine learning models to create workflows for the analysis of protein dynamics and the identification of important protein conformations. Graph embedding effectively captures and compresses structural information from MD simulation data, making it applicable to diverse downstream machine-learning classification tasks.  

An application is presented for encoding and detecting important protein conformations from molecular dynamics simulations to classify functional states. Furthermore, a performance comparison of different graph embedding methods in combination with machine learning models is reported.

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

## **Authors**  

- Ferdoos Hossein Nezhad  
- Namir Oues  
- Massimiliano Meli  
- Alessandro Pandini  

---

## **License**  
This project is licensed under the **GPL-3.0 License**.

