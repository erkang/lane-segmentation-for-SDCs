# Semantic Segmentation
### Introduction
In this project, We are going to label the pixels of a road in a given image using a Fully Convolutional Network (FCN). This project is an extension of Jonathan Long and Evan Shelhamer's work in the paper "Fully Convolutional Network for Semantic Segmentation".

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

##### Dataset
The dataset used for training the model can be downloaded. Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.


##### Implementation

Here I implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are for people that want to get the most out of this project.  They're not required to complete.

##### Run
Run the following command to run the project:
```
python main.py
```

### Solution
The implementation code for the fully convolutional network is available in the ipython notebook. 
