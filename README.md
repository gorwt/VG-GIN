# Enhancing Link Prediction Accuracy with VG-GIN: A Fusion of Variational Graph Auto-Encoders and Graph Isomorphism Networks

This repository provides a PyTorch implementation of VG-GIN (Variational Graph-Graph Isomorphism Network) for link prediction tasks, which has been validated on six benchmark datasets demonstrating superior performance in both AUC and AP metrics compared to baseline models, while supporting multiple graph-structured data input formats (.npy/.npz) .

Note: The code in this repository directly supports the research findings reported in our manuscript submitted to The Visual Computer. This code is shared to ensure reproducibility during the peer review process.

## Requirements
* python>=3.12.0
* torch>=2.5.0 
* networkx>=3.4.0
* scikit-learn>=1.6.0
* scipy>=1.13.0

## Run
* Specify your arguments in `args.py` : you can change dataset and other arguments there
* run `python train_vg_gin.py`
