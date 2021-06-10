# Build a GUI AI Traffic Sign Recognition Project using CNN

you can find explaination of the code on my channel on [Youtube](https://youtu.be/NNSq967DKxE)

The goals / steps of this project are the following:
* Load the data set：[German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
* Explore, summarize and visualize the data set
* Design, train and test a Convenlutional Neural Network model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

## Usage
```learn_signs.py``` for python source code. Additional (```AI_traffic_notebook.ipynb``` for jupyter notebook source code.)
<br>```recognize.py``` for GUI.

## Data Set Summary & Exploration

I  calculate summary statistics of the traffic signs data set:

* The size of training set is 70%
* The size of the validation set is 20%
* The size of test set is 10%
* The shape of a traffic sign image is 30x30x3
* The number of unique classes/labels in the data set is 43

## Model Architecture

Summary for my final model consists of the following layers:
![Model Summary](img/img_1.png)

### Model results

My final model results were:
* Accuracy is 98%

![Model Accuracy](img/img.png)

