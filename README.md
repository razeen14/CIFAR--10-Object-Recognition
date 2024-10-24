# CIFAR-10 Object Recognition using ResNet50

This project implements a deep learning model based on the ResNet50 architecture to classify images from the CIFAR-10 dataset into 10 distinct categories. The CIFAR-10 dataset consists of 50,000 color images of size 32x32 pixels.

# Overview

The goal of this project is to accurately classify images from the CIFAR-10 dataset into one of the following categories:

Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.

We use a pre-trained ResNet50 model, which is fine-tuned on the CIFAR-10 dataset. The project includes data preprocessing, training, evaluation, and visualization of the results.
# Features
Text Preprocessing: Tokenization, stopword removal, and vectorization using TF-IDF (Term Frequency-Inverse Document Frequency).

Classification Model: Logistic Regression is used for binary classification (fake/real).

Performance Metrics: Evaluates model performance using accuracy, precision, recall, and F1 score.

# Dataset
The CIFAR-10 dataset contains 50,000 images of size 32x32 pixels, 10 classes (as listed above). Dateset is split into

Training Set: Used to train the machine learning model.

Test Set: Used to evaluate the performance of the model.

# Workflow

We use ResNet50, a deep convolutional neural network architecture designed for image classification tasks. The model is pre-trained on the ImageNet dataset and then fine-tuned on the CIFAR-10 dataset.

1. Load the CIFAR-10 dataset using tensorflow.keras.datasets .
  
2. Preprocess the data by normalizing pixel values.

3. Use ResNet50 with added layers for classification.
   
4. Train the model on the training set.
   
5. Evaluate the model's performance on the test set.
   
# Installation

1. CLone the repository
   ```bash
   git clone https://github.com/razeen14/CIFAR-10-Object-Recognition.git

   ```
   
2. Navigate to the project directory:
   ```bash
   cd Fake-News-Prediction
   ```
   
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
# Usage

1. Train the model: Run the script to train the model on the provided dataset.
   ```bash
   python train_model.py
   ```

2. Evaluate the model: Evaluate the model's performance on the test data.
   ```bash
   python evaluate_model.py
   ```


