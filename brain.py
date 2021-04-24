import os
from os import listdir
import cv2 as cv
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def contour(image, plot=False):
	'''
	image prepossessing function 
	input:
		image from the dataset
		if plot is needed
	returns:
		new_image: after the processing 
	'''



	img = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #CONVERT THE IMAGE TO GRAYSCALE 
	# thresholding after Gaussian filtering
	blur = cv.GaussianBlur(img,(5,5),0)
	img = cv.threshold(blur,45,255,cv.THRESH_BINARY)[1]
	# opening is erosion followed by dilation. It is useful in removing noise
	
	kernel = np.ones((5,5),np.uint8)
	opening = cv.morphologyEx(img, cv.MORPH_OPEN,None)
    # Find contours in thresholded image, then grab the largest one
	cnts = cv.findContours(img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv.contourArea)
	Left = tuple(c[c[:, :, 0].argmin()][0])
	Right = tuple(c[c[:, :, 0].argmax()][0])
	Top = tuple(c[c[:, :, 1].argmin()][0])
	Bot = tuple(c[c[:, :, 1].argmax()][0])
	new_image = image[Top[1]:Bot[1], Left[0]:Right[0]]

	if plot:
		plt.figure()
		plt.subplot(1,2,1)
		plt.imshow(image)
		plt.title('Original')
		plt.subplot(1,2,2)
		plt.imshow(new_image)
		plt.title('new_image')
		plt.show()
	return new_image


def load(path, img_size):
	x = []
	y=[]
	w, h = img_size
	for directory in path:
		for filename in listdir(directory):
			img = cv.imread(directory + '\\' + filename)
			image = contour(img, plot=False)
			image = cv.resize(image, dsize=(w,h), interpolation=cv2.INTER_CUBIC)
			image = image / 255.
			x.append(image)
			if directory[-3:] == 'yes':
				y.append([1])
			else:
				y.append([0])
    
	x = np.array(x)
	y = np.array(y)
	x, y =shuffle(x,y)
	return x,y

def split(x, y, test_size = 0.1):
	''' 
	This function splits the dataset into three parts: training, validation, and test sets. 
	input from the x and y calculated from the load function
	outputs the separated datasets
	'''
	X_train, X_test_val, y_train, y_test_val = train_test_split(x, y, test_size=test_size)
	X_test, X_val, y_test, y_val = train_test_split(x_test_val, y_test_val, test_size=0.5)
	return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = split(x, y, test_size=0.3)

def model(shape):
	'''
	this function creates the model that used to make the neuronal network that help fit the brain tumor detection
	with the input of the shape of the image 
	and outputs the desired network 
	'''

	X_input = Input(shape)
	X = ZeroPadding2D((2, 2))(X_input) 
	X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
	X = BatchNormalization(axis = 3, name = 'bn0')(X)
	X = Activation('relu')(X)
	X = MaxPooling2D((4, 4), name='max_pool0')(X)
	X = MaxPooling2D((4, 4), name='max_pool1')(X)
	X = MaxPooling2D((4, 4), name='max_pool2')(X)

	X = Flatten()(X)
	X = Dense(1, activation='sigmoid', name='fc')(X) 
	model = Model(inputs = X_input, outputs = X, name='Model')
	model.unfreeze()
	model.lr_find()
	model.recorder.plot()
	history = model.history.history

    
	return model


if __name__ == '__main__':
	path = 'dataset/'
	class1 = path+'yes/'
	class2 = path+'no'
	w, h = (240,240) 
	x,y = load([class1, class2],(w,h))
	shape = (w,h,3)
	model = model(shape)
	model.summary()
	start_time = time.time()
	model.fit(x=X_train, y=y_train, batch_size=32, epochs=100, validation_data=(X_val, y_val))
	end_time = time.time()
	execution_time = (end_time - start_time)
	print(f"Elapsed time: {hms_string(execution_time)}")







