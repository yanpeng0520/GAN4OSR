# Combined-GAN for OSR




# Requirements
- Python 3
- PyTorch 1.10.1
- Cuda 11.4

# Network Architectures
- Small and large version of Combined-GAN
  - Encoder
  - Decoder
  - Discriminator
- LeNet++
- ResNet-18++

# Visualization
- 2D visualization e.g. features from LeNet++
- Bar chart e.g. OSR evaluation results

# Evaluation metrics
- Accuancy
- Confidence
- AUC
- OSRC curve

# Dataset
We use 7 benchmark datasets obtained from PyTorch : MNIST, EMNIST, KMNIST, FMNIST, SVHN, CIFAR-10, CIFAR-100.
We concatenate the 5 datasets in different ways for OSR, for examples, MNIST(KK)+KMNIST(KU) for training and validation, MNIST(KK)+FMNIST(UU) for testing. More detail can be seen in DataManager.py

# Usage
To train and test on MNIST(KK)+ a subset of EMNIST letters (A-M)(KU)
bash excute.sh
