# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Video of Autonomous driving.
![a_driving](https://github.com/unnat5/CarND-Behavioral-Cloning-P3/blob/master/run1.gif)

### Behavioural Cloning Project
The goals/ steps of this project are the following:
- Use the simulator to collect data of good driving behavioural.
- Build, a convolution neural network in Keras that predicts steering angles from images.
- Train and validate the model with a training and validation set.
- Test that the model successfully track one without leaving the road 
- Summarize the results with a written report.


### Project Description
1. **Submission include all required files and can be used to run the [simulator](https://github.com/udacity/CarND-Term1-Starter-Kit) in autonomous mode**


My project includes the following files:
- `model.py` containing the script to create and train the model.
- `drive.py` for driving the car in autonomous mode.
- `model.h5` containing a trained convolution neural network.
- `README.md` summarizing the report.

2. **Submission includes functional code**
<br>Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

3. **Submission code is usable and readable**
<br> The `model.py` file contains the code for training and saving the convolution neural network. The files shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy
1. **An appropriate model architecture has been employed**
| Layer(type)         	|        Description	        				| Output shape | 
|:---------------------:|:---------------------------------------------:|:------------:|
| lambda_1 (Lambda)     | Normalized Image and preprocessed   | (None,100,320,3)|
| conv2d_1 (Conv2D)     | filter=16,kernel_size=8,stride=4,padding="same"| (None,25,80,16)|
| relu_1 (ReLU)     | Activation function | (None,25,80,16)|
| conv2d_2 (Conv2D)     | filter=32,kernel_size=5,stride=2,padding="same"| (None,13,40,32)|
| relu_2 (ReLU)     | Activation function | (None,13,40,32)|
| conv2d_3 (Conv2D)     | filter=64,kernel_size=5,stride=2,padding="same"| (None,7,20,64)|
| relu_3 (ReLU)     | Activation function | (None,7,20,64)|
| flatten_1 (Flatten)     | Flattening layer | (None,8960)|
|dropout_1 (Dropout)|keep_probs=0.2 to reduce overfitting |(None,8960)|
|dense_1 (Dense)|Fully connected layer with 512 nodes|(None,512)|
| relu_4 (ReLU)     | Activation function | (None,512)|
|dense_2 (Dense)|Fully connected layer with 256 nodes|(None,256)|
| relu_5 (ReLU)     | Activation function | (None,256)|
|dense_3 (Dense)|Fully connected layer with 10 nodes|(None,10)|
| relu_6 (ReLU)     | Activation function | (None,10)|
|dense_4 (Dense)|Fully connected layer with 1 nodes|(None,1)|



2. **Attempts to reduce overfitting in the model**
The model contains dropout layer in order to reduce overfitting (model.py line 93)

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 29) with (80-20%) train validation split.

While training I did data augmentation with randomly flipping the training images and accordingly changing the steering angle. These steps were taken to reduce overfitting and to generalize the model better.

One such example is shown below.
<img src = "https://github.com/unnat5/CarND-Behavioral-Cloning-P3/blob/master/examples/img_aug.png">


3. **Model parameter tuning**
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 116).


### Solution design approach

The overall strategy for deriving a model architecture was to treat the problem as a standard computer vision problem I came up with the model architecture on my own the basic idea was to extract image feature with convolution layer and use ReLU as activation function to introduce non-linearity in the above model. In my model I have three conv2D layer for image feature extraction. As the image returned by simulator did not have much complexity to capture so maximum number of filter in my Conv2D layer is 64 and then I used flatten layer to flatten the model and the three fully connected layer to predict steering angle. I have added a Dropout layer with keep_prob=0.2 to reduce overfitting after flatten layer.

### Creation of Training Set and Training Process  
To capture good behaviour, I first recorded two laps on track one I have used mouse to input continuous steering angle. Using mouse helped a lot. The I reverse the car and started driving in opposite direction becuase the track have a bias of taking left turn more than the right turn only 1 right turn and all other are left turns. So to overcome this problem I drove 2 laps in opposite direction. And randomly flipped the images to generalize the model better.

I even used the data which was provided by udacity with the workspace. So my approach was iterative I trained the model with the dataset which I created with the simulator and then save the weights. And then train the model with udacity worksapce dataset and before starting the training process I loaded the previous trained weights to get better results.

I trained the model for 5 epochs with both datasets. 


