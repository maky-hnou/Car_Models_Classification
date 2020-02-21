# Car_Models_Classification  
![Python version][python-version]
[![GitHub issues][issues-image]][issues-url]
[![GitHub forks][fork-image]][fork-url]
[![GitHub Stars][stars-image]][stars-url]
[![License][license-image]][license-url]

Car Models Classifier using TensorFlow.


## About this repo:  
In this repo, I used TensorFlow to build A ResNet50 Neural Network and train it from scratch using the *[Stanford Car Dataset](https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder)*, a dataset containing 196 car model.


## Content:  
- **categories.json:** a json file conaining the car models names.
- **test.py:** the code used to test the model once it is trained.
- **train.py:** the code used to train the model.
- **utils.py:** a python file containing utils functions.
- **resnet50.py:** the code used to build the ResNet50 model.
- **requirements.txt:** a text file containing the needed packages to run the project.
- **main.py:** the file needed to run training, testing and preprocessing.  


## Train and test the model:  

**1. Prepare the environment:**  
*NB: Use python 3+ only.*  
Before anything, please install the requirements by running: `pip3 install -r requirements.txt`.  

**2. Prepare the data:**  
Download the *[Stanford Car Dataset](https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder)*.  
Extract the zip file. It should be organized as follows:  
`data/` should contain a folder named `car_data/`, that contains two folders named `train/` and `test/`.  
Convert the training data to npy file and prepare the labels file by running `python3 main.py` and following the instructions.  

**3. Train and test the ResNet model:** (*from scratch*)
To run both training and testing, you need to run `python3 main.py` then follow the instructions.  


[python-version]:https://img.shields.io/badge/python-3.6+-brightgreen.svg
[issues-image]:https://img.shields.io/github/issues/maky-hnou/Car_Models_Classification.svg
[issues-url]:https://github.com/maky-hnou/Car_Models_Classification/issues
[fork-image]:https://img.shields.io/github/forks/maky-hnou/Car_Models_Classification.svg
[fork-url]:https://github.com/maky-hnou/Car_Models_Classification/network/members
[stars-image]:https://img.shields.io/github/stars/maky-hnou/Car_Models_Classification.svg
[stars-url]:https://github.com/maky-hnou/Car_Models_Classification/stargazers
[license-image]:https://img.shields.io/github/license/maky-hnou/Car_Models_Classification.svg
[license-url]:https://github.com/maky-hnou/Car_Models_Classification/blob/master/LICENSE
