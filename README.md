# PS2

This is a pytorch and pytorch-geometric based implementation of **Bring Your Own View: Graph Neural Networks for Link Prediction with Personalized Subgraph Selection**. 

## Installation

The required packages can be installed by running 
```
pip install -r requirements.txt
```

## Datasets
The datasets used in our paper can be automatically downlowad. 

## Quick Start
Train on the Planetoid datasets (Cora, CiteSeer, and Pubmed):
```
python ps2_planetoid.py --dataset "Cora" 
```