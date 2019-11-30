# ChatSGD

*Experiments on communication-efficient SGD*

# Structure of the repository

The repository contains the following modules:

```
- models.py: contains base class and usable models
- datasets.py: MNIST and CIFAR10 datasets supporting simulated data parallelism
- coding.py: implements QSGD encoding and decoding
- main.py: implements training loop with different settings, and evaluation on classification benchmarks
```

# Installation

TBA

# Running the code

To run an experiment, use `main.py` script:

`python main.py --dataset <mnist, cifar10> --model <lenet> --n_workers 3 --quant_levels [10,10,20] --lr 0.01`