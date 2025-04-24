#!/bin/bash

# Step 2: Merge GraphSAGE embeddings with target state labels
python target_usage.py \
  --embedding embedded_data/graphsage_embeddings.csv \
  --target ADKData/R02_4AKE_DM_prod_T300_dt200_target_value.dat \
  --output target_embedded_data/graphsage5000.csv

# how to run it, first only once run this
# chmod +x run_target.sh  
# Then any time you can run this
# ./run_target.sh 