# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/reduced_brightness_image.png
[image2]: ./examples/left_camera_image.png
[image3]: ./examples/flipped_image.png

---

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. To combat the overfitting, I included dropouts.

The next step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I used RELU activation. But I found ELU gives better results and hence I went with it. 

Also, I augmented the data with cropping, adjusting brightness, resizing, recovering towards center of lane and flipping methods. 

I used generators to generate 4000 training batches per epoch and 1000 validation batches per epoch.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes.

Layer 1: Conv layer with 32 5x5 filters, followed by ELU activation
Layer 2: Conv layer with 16 3x3 filters, ELU activation, Dropout(0.4) and 2x2 max pool
Layer 3: Conv layer with 16 3x3 filters, ELU activation, Dropout(0.4)
Layer 4: Fully connected layer with 1024 neurons, Dropout(0.3) and ELU activation
Layer 5: Fully connected layer with 1024 neurons, Dropout(0.3) and ELU activation

#### 3. Creation of the Training Set & Training ProcessUse left & right camera images to simulate recovery

I used the dataset (nearly about 8000*3 images by center, left and right cameras) provided by Udacity.

After trial and error, I figured out that shaving 55 pixels from the top and 25 pixels from the bottom works well. I also normalized the images.

I reduced the brightness of the image to simulate driving in different lighting conditions.
![alt text][image1]

To simulate the effect of car wandering off to the side, and recovering, I added a small angle .25 to the left camera and subtract a small angle of 0.25 from the right camera. The main idea being the left camera has to move right to get to center, and right camera has to move left.
![alt text][image2]

Since the dataset has a lot more images with the car turning left than right(because there are more left turns in the track), I randomly flipped the images horizontally to simulate turning right and also reversed the corresponding steering angle.
![alt text][image3]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
