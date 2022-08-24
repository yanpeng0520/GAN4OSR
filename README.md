# Combined-GAN for OSR
Overview of the whole model architecture is shown below. The whole model consists of two models: Combined-GAN model (on the top) and open-set model (in the bottom). 
	The generative model has three components: encoder $G_{enc}$ , decoder $G_{dec}$ , discriminator $D$ (including a classifier $C_{cs}$ and an activation function $A$).
	Assume we have image pairs (${x_1}$,${x_2}$) from class "0" and class "5".  From encoder-decoder generative model, reconstructed images from 2 classes and unknown images are generated. Then, $C_{os}$ is trained on the known image samples and generated unknown image samples.
  
<img src="https://github.com/yanpeng0520/GAN4OSR/blob/main/pics/architecture.png" width=80% height=80%>

# Generated samples from MNIST
The figures illustrates two different OSR scenarios. The subfigures on top display two sets of images (separated by a white dash line), which include the original images from one class (1st row), original images from another class (3rd row) and generated images from two different classes (2nd row). The scatter plots in the bottom displays the feature representations of original digit images and generated images using LeNet++. Colored dots represent known samples and different color represents different classes. The black dots represent generated unknown samples. 

<img src="https://github.com/yanpeng0520/GAN4OSR/blob/main/pics/generated_samples.png" width=80% height=80%>

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
- OSRC results (OSCR curve and AUOC)

# Dataset
We use 7 benchmark datasets obtained from PyTorch : MNIST, EMNIST, KMNIST, FMNIST, SVHN, CIFAR-10, CIFAR-100.
We concatenate the 5 datasets in different ways for OSR, for examples, MNIST(KK)+KMNIST(KU) for training and validation, MNIST(KK)+FMNIST(UU) for testing. More detail can be seen in DataManager.py

# Usage
To train and test on MNIST(KK)+ a subset of EMNIST letters (A-M)(KU). 
```
bash excute.sh
```
To generate samples from MNIST and show deep feature visualization using LeNet++. Trained models can be found in model/trained_model.
```
python3 --workers 0 --batchSize 64 --dataset 'mnist' --manualSeed 2 evaluate_sample.py
```
To generate samples from CIFAR-10 and show deep feature visualization using ResNet-18++. Trained models can be found in model/trained_model.
```
python3 --workers 0 --batchSize 64 --dataset 'cifar10' --manualSeed 2 evaluate_sample.py
```


