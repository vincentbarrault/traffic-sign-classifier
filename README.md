# Traffic sign classifier (using Convolutional Neural Network)

This goal of this project is to design a model for a convolutional neural network, train it with a large dataset of german traffic signs (43 different types) and finally test it on personal pictures. All the project was made in Python (using jupyter notebook).

The steps of this project are:
- Load the data set (located in folder '/data')
- Visualize the training data set
- Preprocess all the images from the different data sets (normalizing and converting to grayscale)
-   Design, train, evaluate and test a model architecture based on convolutional neural network.
-   Use the model to make predictions on new images
-   Analyze the softmax probabilities of the new images

## Data Set

### Summary of Data Set

The data sets of german traffic signs consist of:

-   Training set: 34799 images
-   Validation: 4410 images
-   Test set: 12630 images

Every image corresponds to one of the 43 different classes/types of german traffic signs used in this project (see signnames.csv for the list). The shape of an image is 32x32x3 (width, height, depth)

### Visualization of Data Set

The image below shows a histogram of the number of training examples for each class of traffic signs. This was done by using matplotlib. The first classes are mainly speed limit signs and there are a lot of pictures (>1000) for them in this data set. Other classes like "Go straight or right" (36) or "End of no passing" (41) have fewer examples, less than 250 training images for each class. It is important to keep this in mind when evaluating or testing signs, because a training set with few examples of a specific class makes it harder for the neural network to label correctly a traffic sign from this same class.

![enter image description here](https://raw.githubusercontent.com/vincentbarrault/traffic-sign-classifier/master/report_img/Histo_Training_Set.png)

The ratio of images/classes is almost the same for validation and test data sets (not displayed here).

Just as an example, below are 20 pictures choosen randomly in the training data set, with their label on top. Again, displayed by using matplotlib.

![enter image description here](https://raw.githubusercontent.com/vincentbarrault/traffic-sign-classifier/master/report_img/training_set_random.png)

## Preprocess Data Set

Before providing the images as input to the neural network, the data set needs to be preprocessed in order to make the recognition by the algorithm easier and/or faster.

This was done in 2 steps:

- Normalizing images
- Convert them to greyscale

### Normalizing Data Set

The pixel value of an image is between 0 and 255. Normalizing an image can be done in several ways, but one of the easiest way is to convert the pixel values so that their range goes from -1 to 1.

    def normalize_data(set_images):
	    return (set_images-128.0)/128.0

This normalization can help in case of images with a poor constrast, or for removing noise but at the same time bring the image into a range of intensity values that is "normal" (meaning that, statistically, it follows a normal distribution as much as possible)

### Conversion to grayscale

Each image has a depth of 3, corresponding to each color of RGB. Using gray images permits to reduce the quantity of image data to process. It leads also to losing information, but this can be an advantage in some cases (if luminosity is very high on an image, some colors can appear differently, and gray images permit to avoid mistaking a sign by seeing wrong colors).

    def convert_to_grayscale(set_images):
	    return np.sum(set_images/3, axis=3, keepdims=True)

The image below shows how some pictures from the training set look before and after preprocessing.

![enter image description here](https://raw.githubusercontent.com/vincentbarrault/traffic-sign-classifier/master/report_img/normalized_training_set.png)

## Model Architecture

The architecture chosen is very similar to the convolutional neural network (LeNet, Yann LeCun) developed for identifying digits (from 0 to 9).

![enter image description here](https://vitalab.github.io/deep-learning/images/lenet/architecture.png)

For this project, the only differences with the original LeNet architecture are:

- Before performing the first full connection (C5: layer 120), a dropout is performed on the outputs of the previous neural network layer, mainly to overcome the problem of overﬁtting,
- Obviously, the output must correspond to one label out of 43, instead of 10. Consequently, the number of classes is increased to 43.

|  | Layer | Input | Output |
|--|--|--|--|
| 1 | Convolution | 32x32x1 | 28x28x6 |
| 2 | ReLU activation | - | - |
| 3 | Max Pooling  | 28x28x6 | 14x14x6 |
| 4 | Convolution | 14x14x6 | 10x10x16 |
| 5 | ReLU activation |- | - |
| 6 | Max Pooling |10x10x16 | 5x5x16 |
| 7 | Flatten (Input = 5x5x16, Output = 400) |5x5x16 | 400 |
| 8 | Dropout (Probability of 0.5) |- | - |
| 9 | Full connection |400 | 120 |
| 10 | ReLU activation |- | - |
| 11 | Full connection |120 | 84 |
| 12 | ReLU activation |- | - |
| 13 | Full connection |84 | 43 |

The model was trained, then evaluated with a Batch size of **32** and Epoch of **10**:  1 Epoch corresponds to the complete (validation here) data set being passed forward and backward through the neural network.
> Training...
> 
> EPOCH 1 ... Validation Accuracy = 0.827
> 
> EPOCH 2 ... Validation Accuracy = 0.863
> 
> EPOCH 3 ... Validation Accuracy = 0.906
> 
> EPOCH 4 ... Validation Accuracy = 0.912
> 
> EPOCH 5 ... Validation Accuracy = 0.945
> 
> EPOCH 6 ... Validation Accuracy = 0.935
> 
> EPOCH 7 ... Validation Accuracy = 0.940
> 
> EPOCH 8 ... Validation Accuracy = 0.937
> 
> EPOCH 9 ... Validation Accuracy = 0.941
> 
> EPOCH 10 ... Validation Accuracy = 0.951
> 
> Traffic signs model saved

The validation accuracy is **0.951**.

When running the neural network model on the test data set, the test accuracy is **0.937**.

## Testing the Model on New Images

### Using new dataset for testing

New pictures (taken by myself or found on web) are included in this test. They also have a shape of 32x32x3. The same preprocessing as before (normalization, grayscale conversion) is applied on those images.

![enter image description here](https://raw.githubusercontent.com/vincentbarrault/traffic-sign-classifier/master/report_img/normalized_personal_img.png)

Out of the 7 images, 5 are pretty easy to label and 2 are more challenging (low luminosity, picture taken with an angle). As a result, the accuracy on this data set is **0.86** (= 1 image out of 7 is not labelled properly).



### Predictions from CNN

For a given image, the convolutional neural network estimates the probability for each class/label that this image corresponds to it. This probability is given by the softmax function. Let's focus on the first 5 softmax probabilities (= 5 most likely corresponding traffic sign) for each image in order to analyze how well this CNN performs.

#### Traffic sign 1 ('No entry')
Only 2 examples of correctly labelled images with a very high probability (>0.99) are included in this README, but other examples of well recognized traffic signs can be found in the jupyter notebook.

![enter image description here](https://raw.githubusercontent.com/vincentbarrault/traffic-sign-classifier/master/report_img/no_entry.png)

|                |Detected label                         |Probability (%)                    |
|----------------|-------------------------------|-----------------------------|
|1          |No entry          |99.988091           |
|2          |Turn left ahead   |0.010159|
|3          |Stop          |  0.001715          |
|4          |No passing |0.000026|
|5          |Keep right         |0.000006         |

#### Traffic sign 2 ('Keep left')

![enter image description here](https://raw.githubusercontent.com/vincentbarrault/traffic-sign-classifier/master/report_img/keep_left_correct.png)

|                |Detected label                         |Probability (%)                    |
|----------------|-------------------------------|-----------------------------|
|1          |Keep left          |99.997292           |
|2          |Turn right ahead   |0.002626|
|3          |Go straight or left          |    0.000058          |
|4          |Road work |0.000013|
|5          |Yield         |0.000008         |


#### Traffic sign 3 ('Priority road')
For the following picture, the picture was taken in front-view, the only difficulty being the luminosity which is very low. But once the image is converted to grayscale (before being analyzed by the neural network), it is clear that the lack of luminosity is not altering too much the result of the neural network.

![enter image description here](https://raw.githubusercontent.com/vincentbarrault/traffic-sign-classifier/master/report_img/priority_road.png)

|                |Detected label                         |Probability (%)                    |
|----------------|-------------------------------|-----------------------------|
|1          |Priority road          |  98.992470           |
|2          |Roundabout mandatory|0.984287|
|3          |No entry           |0.002345            |
|4          |No passing|0.002298|
|5          |Yield           |0.002164           |



#### Traffic sign 4 ('Wild animals crossing')

The image below is correctly identified, but the softmax probability for the correct label (0.53) is quite low in comparison to other images. The second label is slippery road with a probability of 0.39. This result can be explained by:

- Photo was taken at nightfall, with vehicle lights pointing at it.
- Training data set had about 200 images with the wild animal crossing sign, against more than 1500 training examples for some other signs. This makes the image recognition less robust.

![enter image description here](https://raw.githubusercontent.com/vincentbarrault/traffic-sign-classifier/master/report_img/wild_animal.png)

|                |Detected label                         |Probability (%)                    |
|----------------|-------------------------------|-----------------------------|
|1          |Wild animals crossing            |53.435474           |
|2          |Slippery road|39.066574|
|3          |Dangerous curve to the left           |7.396623            |
|4          |Double curve|0.082238|
|5          |Right-of-way at the next intersection    |  0.018334          |


#### Traffic sign 5 ('Keep left')

The picture below is taken with an angle which makes recognition by the neural network harder. Furthermore, the luminosity of the image is quite low. For these reasons, the detected label (Turn right ahead), with a probability of 0.39, is wrong. The correct label (Keep left) comes only second in the list of softmax probability, with a probability of 0.32, so the neural network outputs are not completely off.

![enter image description here](https://raw.githubusercontent.com/vincentbarrault/traffic-sign-classifier/master/report_img/keep_left_wrong.png)

|                |Detected label                         |Probability (%)                    |
|----------------|-------------------------------|-----------------------------|
|1          |Turn right ahead            |39.327431           |
|2          |Keep left|31.968721|
|3          |Children crossing          |9.236798            |
|4          |Ahead only|9.066751|
|5          |Beware of ice/snow           |5.876662           |

## Possible improvements

Accuracy of **93.7**% is reached on the test data set. Even being satisfying, it is always possible to improve the accuracy of the neural network by following the steps below.

### Improving Model Architecture and Hyperparameters

The architecture of this convolutional neural network is very close to the LeNet-5 architecture. Fine tuning some of the hyperparameters and spending more time training the neural network with different layers might improve the accuracy:  

 - changing the depth of output images for convolution or pooling layers, 
 - performing average or L2-norm pooling, 
 - changing the filter settings for convolution layers like stride or padding, 
 - adding more fully connected layers at the end of the network
 - ...

 ### More data set instances for each class

Some traffic signs are harder to identify because there were not a lot of instances/examples of them in the training set. This could be improved by adding more picture examples of those signs (under various weather conditions, angle...), or by producing more data for each classes by normalizing already existing datas (translation, rotation of existing images):

### Preprocessing data

In this project, simple and quick preprocessing is performed (normalization and grayscale conversion). Normalization could be performed differently, as well as grayscale conversion could be performed using alternative solution, like colorimetric conversion to grayscale. This permits to preserve perceptual luminance by following this formula:
![
](https://wikimedia.org/api/rest_v1/media/math/render/svg/f84d67895d0594a852efb4a5ac421bf45b7ed7a8)
