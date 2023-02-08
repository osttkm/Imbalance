# Imbalance data
This repository contains codes for working on  graduation thesis.　

'src' contains below
1. Data Loader
    * loader for **pyroelectric element** from panasonic
    * loader for **cifar10**
    * loader for **mnist**
2. loss function code
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
