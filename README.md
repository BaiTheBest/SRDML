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

For experiments on any other datasets, one can modify the data input pipeline to fit any given dataset.


If you find this code useful in your research, please consider citing:

    @inproceedings{bai2022saliency,
        title={Saliency-Regularized Deep Multi-Task Learning},
        author={Bai, Guangji and Zhao, Liang},
        booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
        pages={15--25},
        year={2022}
    }

Our [KDD poster](https://github.com/BaiTheBest/SRDML/blob/main/SRDML%20poster.pdf) is also available.
