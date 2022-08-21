# SRDML

This is the code repository for "[Saliency-Regularized Deep Multi-Task Learning](https://dl.acm.org/doi/abs/10.1145/3534678.3539442)" (KDD 2022, Research Track). 

We provide the source code for experiments on both synthetic dataset and CIFAR-MTL.

We test our code with Spyder in Anaconda, Python 3.8, TensorFlow-gpu x2.0 on a Windows machine.

1. Synthetic Experiment

Please directly run synthetic.py.

2. CIFAR-MTL

⋅⋅⋅1) Run generator_cifar10.py to generate CIFAR-MTL dataset we used in our paper.
⋅⋅⋅2) Run base_train.py to train and save the base model, which will be used in SRDML training.
⋅⋅⋅3) Run train.py to train SRDML.
