# The Choquet Integral Neural Network
This is a PyTorch implementation of the paper, "[Enabling Explainable Fusion in Deep Learning with Fuzzy Integral Neural Networks](https://arxiv.org/pdf/1905.04394.pdf)".  If you use this work, please cite as 

    @article{islam2019Enabling,
        author       = {Muhammad Aminul Islam and Derek T. Anderson and Timothy C. Havens and Grant Scott and James M. Keller},
        title        = {{Enabling Explainable Fusion in Deep Learning with Fuzzy Integral Neural Networks}},
        journal={IEEE Transactions on Fuzzy Systems},
        year         = 2019,
        doi          = {10.1109/TFUZZ.2019.2917124},
        publisher    = {IEEE},
        }

This repo consists of the Choquet Integral Neuron module described in the paper and an example that illustrates learning of the Choquet integral from a synthetic dataset. 

## Installation/Dependencies
The code uses [PyTorch](https://pytorch.org/) deep learning frameworks. So, if you haven't it installed on your system, please follow the instructions [here](https://pytorch.org/get-started/locally/). We recommend anaconda as a package manager, which takes care all dependencies.

After installing pytorch and all its dependencies, run the following commands to download and run the example.

```
$ git clone https://github.com/aminb99/choquet-integral-NN.git
$ cd choquet-integral-NN
$ python Choquet_integral_nn_torch.py

```

