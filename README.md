# CIFAR10 Classification with CNN

# Overview:
This project aims to train a CNN model to obtained the correct classification for 32x32 pixel images of 10 types of objects.

# Dataset:
CIFAR10: 60000 32x32 pixel images of the followed classes (6000 for every class):
airplane; automobile; bird; cat; deer; dog; frog; horse; ship; truck.

# Methods:
The model is a CNN that, after some tries, is made of:
- Data Augmentation before the input layers: RandomHorizontalFlip, RandomCrop and Normalization with means;
- A first Convolutional layer that increase the number of channels to 32, followed by a BatchNormalization and Relu activation;
- a first Pool to reduce images to 16x16;
- A second Convolutional layer that increase the number of channels to 64, followed by a BatchNormalization and Relu activation;
- a second Pool to reduce images to 8x8;
- A third Convolutional layer that increase the number of channels to 128, followed by a BatchNormalization and Relu activation;
- a third Pool to reduce images to 4x4;
- a Dropout and a linear layer.

The number of parameters of this model is approximately 94144.
It was used cudnn.benchmark because the input shape is fixed.

# Results:
The model obtained 0.79 accuracy and a good f1-score in every class, except for class 3 (cats) that got a bad recall.
