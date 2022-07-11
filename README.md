# L2O-with-GNN
Learn to Optimize Neural Networks using meta-trained GNN

## Objective
Learn to Optimize is a paradigm that aims at meta-training an optimizer to train other optimizees, for example, neural networks. Current implementations use either RNNs or reinforcement learning, and they output the update to parameters, given the current gradient.

In this project, we will try to integrate graph learning in learned optimizers to train neural networks.

## Project Files

`main.ipynb` is a notebook that compares different training the same neural network with different optimization methods. The neural network is a MNIST classifier with 32 units in its hidden layer, and uses ReLU as its activation function.
