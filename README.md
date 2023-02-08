# Imbalance data
This repository contains codes Oshita used to work on  graduation thesis.　

'src' contains below
1. Data Loader
    * loader of **pyroelectric element** from panasonic
    * loader of **cifar10**
    * loader of **mnist**
2. losses.py contains functions below
    * **LDAM Loss** <sup>
    <sup>[1](https://arxiv.org/abs/1906.07413)
    * **Focal Loss** <sup>
    <sup>[2](https://arxiv.org/abs/1708.02002)
    * other

### Main usage
Make running cb_train.py .
### Experiment Manager
MLFLOW
### Environment
* python3.9
* cudatoolkit 11.3

## Coution
In this experiment, seed is realy important. 

Make sure checking seed by seed_check.py
