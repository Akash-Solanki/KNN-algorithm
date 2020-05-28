# k-nearest-neighbors

Implementation of KNN algorithm in MATLAB

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
* Attributes can be integer or real values.
* List attributes first, and add response as the last parameter in each row.
    * E.g. [4.5, 7, 2.6, "Orange"], where the first 3 numbers are values of attributes and "Orange" is one of the response classes.
    * Another example can be [1.2, 4.3, 3], in this case there are 2 attributes while the response class is the integer 3.
    * The square brackets are shown for convenience in reading, don't put them in your CSV file.
* Responses can be integer, real or categorical.

## Using provided data sets

* The Iris data set is provided in the repository. 
* Enter 'iris-dataset.csv' when asked for training data file name.
* Enter 'iris-test.csv' when asked for test data file name.

## Notes

* Keep the data set files in the working directory of project as defined by the IDE configuration.
* When running in stand alone mode (E.g. command line), keep the data sets in the same directory as the script.



