# L2O-with-GNN
This project implements a learned optimizer for neural networks using graph recurrent neural networks. The results are not as satisfactory as originally wished. Any new major changes or advances in the research will probably result in a new github repo with a drastically different pipeline/model architecture.

This repo can still be used as a reference for myself and other researchers.

This project is a part of the 2022 Summer Research Internship held by CUHK's Faculty of Engineering.

## Objective
Learn to Optimize is a paradigm that aims at meta-training an optimizer to train other optimizees, for example, neural networks. Current implementations use either RNNs or reinforcement learning, and they output the update to parameters, given the current gradient.

In this project, we will try to integrate graph learning in learned optimizers to train neural networks.

## Project Files

`metatrain_lstm.ipynb` is a notebook which trains a l2o optimizer. This optimizer is based on a paper by Marcin Andrychowicz et al.

`baseline_comp.ipynb` trains neural networks using traditional methods (SGD, Adam and RMSprop) and l2o methods and compare the results.

`systematic_comp.ipynb` trains 10 LSTM optimizers and 10 GNN optimizers, and compares the statistics of their performance.
