# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from getProcessedImages import getProcessedImages

######## Part 1 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))


# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

######## Part 2 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set

test_set = getProcessedImages()
training_set = getProcessedImages(isTest=False)

cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

######## Part 3 - Making a single prediction

import numpy as np
from keras.preprocessing import image
test_image = image.load_img(
    'single_prediction/w_or_b_1.JPG',
     target_size = (64, 64))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
print(training_set.class_indices)
if result[0][0] == 1:
    prediction = 'bird'
else :
    prediction = 'windsurfer'

print(prediction)

test_image = image.load_img(
    'single_prediction/w_or_b_2.jpg',
    target_size=(64, 64))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
print(training_set.class_indices)
if result[0][0] == 1:
    prediction = 'bird'
else:
    prediction = 'windsurfer'

print(prediction)
