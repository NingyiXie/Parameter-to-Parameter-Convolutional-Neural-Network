# Parameter-to-Parameter-Convolutional-Neural-Network
This repository is implementation of the "Quantum Approximate Optimization Algorithm Parameter Prediction Using a Convolutional Neural Network"

## Introduction
We propose a CNN model named PPN to learn mappings between parameters from different depths QAOA.  

We provide two trained PPNs in [models](models/): the first is trained with 8-node Erdős–Rényi graphs with 0.5 edge probability and the second is trained with multi-types graphs.  

And we offer two strategies based on the proposed model, which are implemented in `strategies.py`.  

## Requirements
* Qiskit 0.32.1
* PyTorch 1.9.0
* Numpy 1.21.4
* qaoalib 0.1.6
* Scipy 1.7.3
* NetworkX 2.6.3
