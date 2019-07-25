## Pre-processing and library importing.
import csv 
import cv2
import matplotlib.image as mpimg
import numpy as np
import os
import tensorflow as tf
current_dir = os.getcwd()
lines = []

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras import backend as K
from keras.layers import Input, Lambda
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split


## Extracting data from csv file.
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)


## Train-validation split (80-20%)
train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)

## Creating the generator.
def generator(lines, batch_size):
    num_samples = len(lines)
    while True:
        sklearn.utils.shuffle(lines)
        for offset in range(0,num_samples,batch_size):
            batch_samples = lines[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = current_dir + '/my_data/IMG/'+batch_sample[0].split('/')[-1]
                left_name = current_dir + '/my_data/IMG/'+batch_sample[1].split('/')[-1]
                right_name = current_dir + '/my_data/IMG/'+batch_sample[2].split('/')[-1]
                center_image = mpimg.imread(center_name)
                left_image = mpimg.imread(left_name)
                right_image = mpimg.imread(right_name)
                center_angle = float(batch_sample[3])
                left_angle = center_angle + 0.2
                right_angle = center_angle - 0.2
                ### Flipping the image.
                left_flip = np.fliplr(left_image)
                right_flip = np.fliplr(right_image)
                lf_angle = (-1.0)*left_angle
                rf_angle = (-1.0)*right_angle
                ## array subsetting is use to crop the image to only show road lanes.
                images.extend([center_image[60:,:],left_image[60:,:],right_image[60:,:],left_flip[60:,:],right_flip[60:,:]])
                angles.extend([center_angle,left_angle,right_angle,lf_angle,rf_angle])
                
        X_train = np.array(images)
        y_train = np.array(angles)
        yield sklearn.utils.shuffle(X_train, y_train)
        
        
train_generator = generator(train_samples,512)
validation_generator = generator(validation_samples,512)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, ReLU,Conv2D

model = Sequential()
## Normalizing the the incoming image data.
model.add(Lambda(lambda x: x/127.5 -1.,
         input_shape = (100,320,3),
         output_shape = (100,320,3)))
## First convolution layer with 16 filters and kernel size = 8, stride = 4
## larger stride is used to downsample the image.
model.add(Conv2D(16, kernel_size= (8, 8), strides=(4, 4), padding="same"))
## Added an activation layer for adding nonlinearity.
model.add(ReLU())
## Second convolution layer with 32 filters and kernel size = 5, stride =2 
## stride = 2, again downsample the image.
model.add(Conv2D(32, kernel_size= (5, 5), strides=(2, 2), padding="same"))
## Added an activation layer for adding nonlinearity.
model.add(ReLU())
## Third convolution layer with 64 fulter and kernel size = 5, stride =2
## stride = 2, again downsample the image.
model.add(Conv2D(64, kernel_size= (5, 5), strides=(2, 2), padding="same"))
## Added an activation layer for adding nonlinearity.
model.add(ReLU())
## Flattening the model so could connect the dense layer.
model.add(Flatten())
## Added a dropout layer to avoid overfitting after flattening of the layer.
model.add(Dropout(.2))
## Added a dense(fully connected) layer with 512 nodes
model.add(Dense(512))
## Added an activation layer for adding nonlinearity.
model.add(ReLU())
## Added a dense(fully connected) layer with 256 nodes
model.add(Dense(256))
## Added an activation layer for adding nonlinearity.
model.add(ReLU())
## Added a dense(fully connected) layer with 10 nodes
model.add(Dense(10))
## Added an activation layer for adding nonlinearity.
model.add(ReLU())
## Added a dense (fully connected) layer with 1 node as model is prediciting steering angle.
model.add(Dense(1))


# from keras.models import load_model
from keras.models import load_model
## Iteratively training my model.With loading the previous trained weight and then
## train on new dataset.
model = load_model('model.h5')
## Loss used in my mode is mean-squre error and 'Adam' optimizer.
model.compile(loss = 'mse',optimizer='adam')
## I found out having a higher batch_size helped me in optimizing my model faster
## in my above model architecture.

batch_size = 512
## In my case I have 80-20% train-validation split to reduce overfitting.
## Used genertor to load data so all the data is not stored in Ram and large 
## dataset can cause memory overflow is it's a good practice to use generator.
model.fit_generator(train_generator, \
                   steps_per_epoch=len(train_samples)//batch_size,\
                   validation_data = validation_generator, \
                   validation_steps = len(validation_samples)//batch_size,\
                   epochs=4,verbose=1)
## Save the model 
model.save('model.h5')
