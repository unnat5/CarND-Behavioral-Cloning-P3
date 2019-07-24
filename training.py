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
from keras.applications.inception_v3 import InceptionV3,preprocess_input


with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
import sklearn
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)
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
                images.extend([center_image[60:,:],left_image[60:,:],right_image[60:,:],left_flip[60:,:],right_flip[60:,:]])
                angles.extend([center_angle,left_angle,right_angle,lf_angle,rf_angle])
                
        X_train = np.array(images)
        y_train = np.array(angles)
        yield sklearn.utils.shuffle(X_train, y_train)
        
        
train_generator = generator(train_samples,512)
validation_generator = generator(validation_samples,512)
images = []
measurements =[]
for line in lines[1:]:
    center_source_path  = line[0]
    left_source_path = line[1]
    right_source_path = line[2]
    center_filename = center_source_path.split('/')[-1]
    left_filename = left_source_path.split('/')[-1]
    right_filename = right_source_path.split('/')[-1]
# 	print(source_path,filename)
    center_current_path = current_dir+'/data/IMG/'+center_filename
    left_current_path = current_dir+'/data/IMG/'+left_filename
    right_current_path = current_dir+'/data/IMG/'+right_filename
# 	print(current_path)
    center_image  = mpimg.imread(center_current_path)
    left_image = mpimg.imread(left_current_path)
    right_image = mpimg.imread(right_current_path)
    images.extend([center_image[60:,:],left_image[60:,:],right_image[60:,:]])
    images.extend([np.fliplr(left_image[60:,:]),np.fliplr(right_image[60:,:])])
    
    
    correction = 0.3
    
    measurement = float(line[3])
    measurements.extend([measurement,measurement-0.2,measurement+0.2])
    measurements.extend([(-1.0)*(measurement-0.2),(-1.0)*(measurement+0.2)])

X_train = np.array(images)
Y_train = np.array(measurements)


 
# def resize(image):
#     import tensorflow as tf 
#     from keras.applications.inception_v3 import preprocess_input
#     image =preprocess_input(image)
#     image = (image/127.5)- 1.
#     out = tf.image.resize_images(image, (139, 139)) 
#     return out
# from keras.layers import Cropping2D
# input_ = Input(shape=(100,320,3))
# # input_ = Cropping2D(cropping = ((50,20), (0,0)) )(input_)
# # print(input_)
# input_size = 139
# # Re-sizes the input with Keras's Lambda layer and attach to cifar_input
# resized_input = Lambda(resize)(input_)

# ## create the base pre-trained model
# base_model = InceptionV3(input_tensor =resized_input ,weights='imagenet',include_top=False)

# ## add a global spatial average pooling layer
# x = base_model.output
# x = Flatten()(x)
# x = Dropout(0.2)(x)
# # x = GlobalAveragePooling2D()(x)
# # lets add a fully connected layer
# x = Dense(512,activation='elu')(x)
# # and a regression model.
# # x = Dense(64,activation='elu')(x)
# x = Dropout(0.5)(x)
# # x = Dense(10,activation='elu')(x)
# predictions = Dense(1,activation=None)(x)



# # this is the model we will train
# model=Model(inputs=base_model.input,outputs=predictions)


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
# for layer in base_model.layers:
#     layer.trainable = False


# model.load("model1.h5")

# model.load_weights('umodel.h5')
# model.save('new_model.h5')
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, ReLU,Conv2D
model = Sequential()
model.add(Lambda(lambda x: x/127.5 -1.,
         input_shape = (100,320,3),
         output_shape = (100,320,3)))

model.add(Conv2D(16, kernel_size= (8, 8), strides=(4, 4), padding="same"))
model.add(ReLU())
model.add(Conv2D(32, kernel_size= (5, 5), strides=(2, 2), padding="same"))
model.add(ReLU())
model.add(Conv2D(64, kernel_size= (5, 5), strides=(2, 2), padding="same"))
model.add(ReLU())
model.add(Flatten())
model.add(Dropout(.2))
model.add(Dense(512))
model.add(ReLU())
model.add(Dense(256))
model.add(ReLU())
model.add(Dense(10))
model.add(ReLU())
model.add(Dense(1))


# from keras.models import load_model
# model = load_model('umodel.h5')

from keras.models import load_model
# model = load_model('model.h5')
model.compile(loss = 'mse',optimizer='adam')
# model.fit_generator(train_generator,steps_per_epoch = len(lines)//512,epochs=5)
batch_size = 512
# model.fit_generator(train_generator, \
#                    steps_per_epoch=len(train_samples)//batch_size,\
#                    validation_data = validation_generator, \
#                    validation_steps = len(validation_samples)//batch_size,\
#                    epochs=4,verbose=1)
model.fit(X_train,Y_train,shuffle=True,nb_epoch=2,batch_size=512)
model.save('model.h5')
