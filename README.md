# ECBM4040 Course Project Code

Student: Aaron Geelon So

Project Report:
[Considering an Information-Theoretic Foundation to Question of
Generalization in Neural
Networks](https://geelon.github.io/projects/files/2017-fall/neural-networks-final-report.pdf) 


## Overview

This project was inspired by [Understanding Deep Learning Requires
Rethinking Generalization](https://arxiv.org/pdf/1611.03530.pdf), by
Zhang, et. al. 2016. In short, neural networks are able to achieve
high accuracy on most datasets---this includes shuffling the labels or
even assigning random labels to an existing dataset. Even worse, high
accuracy may be achieved on random datasets, even with
regularization. This suggests that we cannot attribute the
generalizability of neural networks only to regularization; we should
develop further theory to understand the phenomenon of
generalizability. Perhaps that will allow us to take a more scientific
approach to training future neural networks.


## Files

- [generalization.ipynb](./generalization.ipynb): Generates the output
  for the project.
- [cifar_utils.py](./cifar_utils.py): Downloads and generates batches
  of CIFAR data.
- [data_management.py](./data_management.py): Generates batches of
  training data.
- [default.py](./default.py): The default network architecture
  parameters.
- [learner.py](./learner.py): Creates learning model and method to
  test generalization error.
- [preprocess.py](./preprocess.py): Preprocesses CIFAR data for
  learning model.

Please see jupyter notebook and code for more documentation.