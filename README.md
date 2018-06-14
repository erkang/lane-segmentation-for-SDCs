# Lane Segmentation for Self Driving Cars using Fully Convolutional Networks
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

Here I implement the code in the `main.py` module to develop the model using the layers of VGG Network using a skip layer architecture and then write optimization and training code for this model. And then this model is used to produce results using the test data of the `Kitti Road Dataset`.

##### Run
Run the following command to run the project:
```
python main.py
```

### Solution
The implementation code for the fully convolutional network is available in the ipython notebook as well. You can use the jupyter notebook to study the code using a cell by cell approach to see what is happening under the hood.

### Results
Here are some cherry picked examples from the results produced.

When the following images go in:

![image1](https://raw.githubusercontent.com/rtspeaks360/lane-segmentation-for-SDCs/master/imgs/Screen%20Shot%202018-06-14%20at%205.18.11%20AM.png)

![image2](https://raw.githubusercontent.com/rtspeaks360/lane-segmentation-for-SDCs/master/imgs/Screen%20Shot%202018-06-14%20at%205.17.47%20AM.png)

![image3](https://raw.githubusercontent.com/rtspeaks360/lane-segmentation-for-SDCs/master/imgs/Screen%20Shot%202018-06-14%20at%205.18.47%20AM.png)

Here are the results that we get.

![image1-result](https://raw.githubusercontent.com/rtspeaks360/lane-segmentation-for-SDCs/master/imgs/Screen%20Shot%202018-06-14%20at%205.18.24%20AM.png)

![image2-result](https://raw.githubusercontent.com/rtspeaks360/lane-segmentation-for-SDCs/master/imgs/Screen%20Shot%202018-06-14%20at%205.17.58%20AM.png)

![image3-result](https://raw.githubusercontent.com/rtspeaks360/lane-segmentation-for-SDCs/master/imgs/Screen%20Shot%202018-06-14%20at%205.18.56%20AM.png)

You can also produce the same results yourself using the code here. The same model can also be used to process video files using minor modifiations to the code base.
