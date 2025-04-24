#!/bin/bash
# Step 1: Generate GraphSAGE embeddings from MD data
python embedding_usage.py \
  --trajectory ADKData/R02_4AKE_DM_prod_T300_protein_dt200.xtc \
  --topology ADKData/R02_4AKE_DM_prod_T300_protein.gro \
  --output embedded_data/graphsage_embeddings.csv

# how to run it, first only once run this
# chmod +x run_embedding.sh  
# Then any time you can run this
# ./run_embedding.sh        

