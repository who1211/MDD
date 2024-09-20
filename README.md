## Major depressive disorde; Rs-fMRI;  
## Code for APO-GCN model
Diagnosis of major depressive disorder using a novel interpretable GCN model based on resting state fMRI 
An adaptive propagation operator graph convolutional network

## Overview
The propagation operator of the APO-GCN, incorporating trainable channel weighting factors (ω), substantially augments the adaptability of the model.
Figure [1]
![image](https://github.com/user-attachments/assets/42a6f788-e5bf-4922-96ba-519256998046)

## Code Description
the main architecture of APO-GCN lies in the `models.py`. The `APO-GCN1 and APO-GCN2.py` are the main backbone, while the rest necessary modules are distributed into different files based on their own functions, i.e., `VariableOperator.py`, `Variable_chevconv.py`. Please refer to each file to acquire more implementation details. 
## Environment config
numpy                     1.26.0
python                    3.9.18
tensorflow                2.10.0
torch                     1.12.0+cu116
torch-geometric           2.4.0
torchmetrics              1.3.2  
torchvision               0.13.0+cu116
