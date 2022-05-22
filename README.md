# Combined-GAN for OSR
The whole model consists of two models: Combined-GAN model (on the top) and open-set model (in the bottom). 
	The generative model has three components: encoder $G_{enc}$ , decoder $G_{dec}$ , discriminator $D$ (including a closed-set classifier $C_{cs}$ and an activation function $A$).
	Assume we have image pairs (${x_1},{x_2}$) from class "0" and class "5".  From encoder-decoder generative model, reconstructed images from 2 classes and unknown images are generated. Then, $C_{os}$ is trained on the known image samples and generated unknown image samples.
  
![Overview of the whole model architecture](./test/arch_0513.png?raw=true "Overview of the whole model architecture"){:height="50%" width="50%"}

# Generated samples from MNIST
The figure illustrates two different OSR scenarios. The figures on top display two sets of images (separated by a white dash line), which include the original images from one class (1st row), original images from another class (3rd row) and generated images from two different classes (2nd row). The scatter plots in the bottom displays the feature representations of original digit images and generated images using LeNet++. Colored dots represent known samples and different color represents different classes. The black dots represent generated unknown samples. 
![Alt text](./test/g.png?raw=true "Title")

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
To train and test on MNIST(KK)+ a subset of EMNIST letters (A-M)(KU). 
```
bash excute.sh
```
