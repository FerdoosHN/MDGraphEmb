#!/bin/bash

# Step 3: Train and evaluate ML models on labeled embeddings and 
# You can choose you classfier here I only inulcded 4, you can also add "Neural Network", "CNN","Support Vector".
python ml_usage.py \
  --data target_embedded_data/graphsage5000.csv \
  --report report \
  --classifiers "Logistic Regression" "Random Forest" "XGBoost" "LightGBM"

# how to run it, first only once run this
# chmod +x run_ml.sh  
# Then any time you can run this
# ./run_ml.sh 