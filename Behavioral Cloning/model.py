#importing packages
#basic packages
import numpy as np
import pandas as pd 
from sklearn.utils import shuffle
import math
#Image processing utilities 
import matplotlib
import matplotlib.image as mpimg
from PIL import Image
# helper functions from Keras
import keras 
from keras.models import Sequential
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.optimizers import Adam


#helper function to crop images
def crop_image(img):
        width = img.size[0]
        height = img.size[1]
        cropped_img = img.crop((0 ,60,width,height-35))
        return cropped_img

#gernerator function for training dataset    
def training_genarator(batch_size, train_data) : 
    directory_center=train_data['center'] # directory of images from central camera
    y_steering=train_data['steering angle'] # steering angle data
    directory_left=train_data['left'] # directory of images from left camera
    directory_right=train_data['right'] # directory of images from right camera
    steer_left=y_steering+0.2 #left steer
    steer_right=y_steering-0.2 #right steer
    #concatenating entire dataset
    directory=np.concatenate([directory_center,directory_left,directory_right])
    sterring_angle=np.concatenate([y_steering,steer_left, steer_right])
    directory, sterring_angle = shuffle(directory, sterring_angle) #shuffling data 
    batch_features = np.zeros((batch_size, 65, 320, 3)) # according to the crop size 
    batch_labels = np.zeros((batch_size,1))
    
    while True:
        temp=0
        d=np.random.permutation(len(directory))
        for i in range(batch_size):
            index=d[i+temp]
            input_image=Image.open(directory[index].strip()) #opening image using PIL
            input_image=crop_image(input_image) # cropping image
            i_np=np.asarray(input_image) #coverting it into a numpy array
            i_np=(i_np/255.0) -0.5 #normalising the input image            
            batch_features[i] = i_np #input feature
            batch_labels[i] = sterring_angle[index]  #input steering angle           
            flip_random = np.random.randint(0,1)
            if flip_random == 1:
                batch_features[i] =np.fliplr(i_np)
                batch_labels[i] = -sterring_angle[index] 
        temp=temp+batch_size    
        yield batch_features, batch_labels 

#validation data generator

def validation_genarator(batch_size, valid_data) : 
    # data=pd.read_csv('/Users/Enkay/Documents/Viky/python/self-driving/behaviorialCloning/driving_log.csv',
    #             names=['center','left','right','steering angle', 'throttle', 'brake', 'speed'])
    directory_center=valid_data['center']
    y_steering=valid_data['steering angle']
    directory_left=valid_data['left']
    directory_right=valid_data['right']
    steer_left=y_steering+0.2
    steer_right=y_steering-0.2
    directory=np.concatenate([directory_center,directory_left,directory_right])
    sterring_angle=np.concatenate([y_steering,steer_left, steer_right])
    directory, sterring_angle = shuffle(directory, sterring_angle)
    batch_features = np.zeros((batch_size, 65, 320, 3))
    batch_labels = np.zeros((batch_size,1))
    
    while True:
        temp=0
        d=np.random.permutation(len(directory))
        for i in range(batch_size):
            index=d[i+temp]
            input_image=Image.open(directory[index].strip())
            input_image=crop_image(input_image)
            i_np=np.asarray(input_image)
            i_np=(i_np/255.0)-0.5 #normalising the input image
            #i_np=matplotlib.colors.rgb_to_hsv(i_np)
            batch_features[i] = i_np
            batch_labels[i] = sterring_angle[index] 
        yield batch_features, batch_labels 

# No. of samples and splitting data to train/validation
data=pd.read_csv('/Users/Enkay/Documents/Viky/python/self-driving/behaviorialCloning/driving_log.csv',
                 names=['center','left','right','steering angle', 'throttle', 'brake', 'speed'])

batch_size=10
data=data.iloc[np.random.permutation(len(data))]
data=data.reset_index(drop=True)
training_data=data[0:math.ceil(len(data)*0.8)]
validation_data=data[math.ceil(len(data)*0.8):len(data)]
no_train_samples=len(training_data)*3
no_val_samples=len(validation_data)
#Model architecture
model = Sequential()
# First convolutional layer
model.add(Convolution2D(32, 5, 5, border_mode='same',  W_regularizer = l2(0.001), input_shape=(65,320,3)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
# Second convolutional layer
model.add(Convolution2D(64, 5, 5, border_mode='same',  W_regularizer = l2(0.001)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
# Third convolutional layer
model.add(Convolution2D(64, 5, 5, border_mode='same',  W_regularizer = l2(0.001)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
# Fourth convolutional layer
model.add(Convolution2D(64, 5, 5, border_mode='same' ,  W_regularizer = l2(0.001)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
#Flatteing layer to feed to fully connected layers
model.add(Flatten())
#fully connected layer 1
model.add(Dense(128))
model.add(Dropout(.5))
#fully connected layer 2
model.add(Dense(64))
model.add(Dropout(.5))
#fully connected layer 3
model.add(Dense(32))
model.add(Dropout(.5))
#Output layer
model.add(Dense(1))
#Optimiser
adam=Adam(lr=0.0001)
model.compile(optimizer= adam, loss='mse')
model.fit_generator(training_genarator(batch_size,training_data), samples_per_epoch = no_train_samples, nb_epoch = 10,
                    verbose=2, show_accuracy=True, callbacks=[], 
                    validation_data=validation_genarator(batch_size, validation_data), nb_val_samples=no_val_samples,
                    class_weight=None, nb_worker=1)

#saving the model 
model.save('model.h5')