# L2O-with-GNN
Learn to Optimize Neural Networks using meta-trained GNN

## Objective
Learn to Optimize is a paradigm that aims at meta-training an optimizer to train other optimizees, for example, neural networks. Current implementations use either RNNs or reinforcement learning, and they output the update to parameters, given the current gradient.

In this project, we will try to integrate graph learning in learned optimizers to train neural networks.

## Project Files

`main.ipynb` is a notebook that compares different training the same neural network with different optimization methods. The neural network is a MNIST classifier with 32 units in its hidden layer, and uses ReLU as its activation function.

`metatrain.ipynb` meta-trains the selected optimizer. 

## Meta-Training

Training the L2O optimizer is known as Meta-Training. First, a forward pass is done with the optimizee, which calculated a batch of parameters with gradients. The batch is passed to the optimizer and outputs a prediction. The prediction is used to update the optimizee. The optimizee then does a forward pass again, which comes with a calcualted loss with respect to the optimizer. The loss is used to update the optimizer via backprop.
