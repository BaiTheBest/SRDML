# SRDML

This is the code repository for "[Saliency-Regularized Deep Multi-Task Learning](https://dl.acm.org/doi/abs/10.1145/3534678.3539442)" (KDD 2022, Research Track). 

We provide the source code for experiments on both synthetic dataset and CIFAR-MTL.

We test our code with Spyder in Anaconda, Python 3.8, TensorFlow-gpu x2.0 on a Windows machine.

1. Synthetic Experiment

   - Run synthetic_train.py.

2. CIFAR-MTL

   - Run generator_cifar10.py to generate CIFAR-MTL dataset we used in our paper.
   - Run pretrain.py for the pre-training to get the pre-trained model.
   - Run train.py.







