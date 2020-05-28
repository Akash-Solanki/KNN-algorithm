# k-nearest-neighbors

Implementation of KNN algorithm in python 3

## Description

* K-Nearest-Neighbors algorithm is used for classification and regression problems.
* In this project, it is used for classification.
* To classify the grayscale images of digits (0-9) from MNIST Dataset using vector as pixels of the image or histogram of the image  and   distance metric like euclidean distance using 1,3,5,9,11 nearest neighbour classifier 

## Data set format

* MNIST Dataset:
  * MNIST (Modified National Institute of Standards and Technology) database is a
   large database for handwritten digits. Download the MNIST files. For both train 
   as well as test data the first digit in the ith row
   is the label (a number in the range 0-9) of the ith sample. The next 784 (28x28)
   digits in the same row are the values of the pixels of the sample (image). The
   image is stored in row-major order, so that the first 28 entries of the 784 digits
   are the pixel values of the first row of the image.
* CIFAR10 Dataset:
  * In both training as well as in test data each row stores a 32x32 color image. The
    first 1024 entries contain the red channel values, the next 1024 the green, and
    the final 1024 the blue. The image is stored in row-major order, so that the first
    32 entries of the array are the red channel values of the first row of the image.
    When you load the training or testing data the corresponding labels will be
    automatically loaded.
* Attributes are real integer values.

## Using provided data sets

* The MNIST and CIFAR10 is provided in the repository. 
* Enter 'iris-dataset.csv' when asked for training data file name.
* Enter 'iris-test.csv' when asked for test data file name.

## Notes

* Keep the data set files in the working directory of project as defined by the IDE configuration.
* When running in stand alone mode (E.g. command line), keep the data sets in the same directory as the script.



