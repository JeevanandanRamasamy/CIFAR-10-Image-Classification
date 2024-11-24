# CIFAR-10 Image Classification

This repository demonstrates the implementation of a custom deep learning framework for training neural networks on the CIFAR-10 dataset. The code covers various aspects of machine learning, from data preprocessing to defining and training models, including fully connected networks (FC) and convolutional neural networks (CNN).

## Overview
- **Dataset**: CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 50,000 training images and 10,000 test images.
- **Model Types**:
  - Two-Layer Fully Connected Network
  - Three-Layer Convolutional Network
  - Custom Linear Layer Implementation
  - Custom Convolution Layer Implementation
  - Sequential Models with Batch Normalization, Dropout, and MaxPooling
- **Training**: Implements stochastic gradient descent (SGD) with weight decay, momentum, and learning rate decay.
- **Evaluation**: Accuracy is measured on both training and validation sets.

---

## Dependencies
- `torch`: PyTorch deep learning framework.
- `torchvision`: Datasets and transformations for image data.
- `numpy`: For numerical operations.
- `matplotlib`: For plotting and visualizing results.

To install the required packages, run:
```bash
pip install torch torchvision numpy matplotlib
```

## File Structure

**Data Preprocessing**:
- The code begins by loading the CIFAR-10 dataset and normalizing the images using the torchvision.transforms module.
- The dataset is split into training, validation, and test sets.
  
**Model Definitions**:
-	TwoLayerFC: A simple fully connected neural network.
-	ThreeLayerConvNet: A convolutional neural network with two convolutional layers followed by a fully connected layer.
-	Custom Layers: Implements custom linear and convolutional layers with explicit forward propagation.

**Training and Evaluation**:
-	Training is performed using train_part34, with optimization techniques such as SGD and learning rate decay.
-	Accuracy is evaluated on the validation set after each epoch, and final testing accuracy is computed on the test set.

---

## Results

The final model achieves over 70% accuracy on the CIFAR-10 test set after training with SGD and regularization techniques like batch normalization, dropout, and data augmentation. Although this is not a significantly high accuracy, the training of the model is extremely fast and efficient. This is beneficial because it demonstrates that the model can achieve reasonable performance with efficient training, making it suitable for quick experimentation and deployment in real-world scenarios where speed and resource efficiency are crucial.


