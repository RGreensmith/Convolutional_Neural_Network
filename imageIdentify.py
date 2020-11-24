import numpy as np
from keras.preprocessing import image
from os import listdir
import re

def isImage (fileName):

    """
    Uses a regular expression to check whether a file is an image (jpg|png|bmp|gif).

    Returns:
        Boolean: Is the file an image?
    """

    return re.match(".*\.(jpg|png|bmp|gif)$", fileName.lower()) != None


def imageIdentify (imageDir,cnn,trainingSet):

    """
    1) Creates a list of file names for images within the imageDir.
    2) For each image file name:
        2.1) Loads image.
        2.2) Predicts the identity of each image using your Convolutional Neural Network (CNN).
        2.3) Uses the file name from your CNN training set to state the prediction.
        2.4) Uses the string of file names of the images you want to identify to reference the predictions
            of identity.

    Args:
        imageDir (String): Pathway to folder containing images you want the CNN to identify.

        cnn (CNN object of keras Sequential): Trained CNN object you want to use to identify images.

        trainingSet (dataset object of keras ImageDataGenerator): Dataset used to train the CNN.

    Returns:
        String: The CNN's prediction of the identity of your images,
                with a dictionary of the file names of the images for reference.
    """

    imageNames = list(filter(isImage, listdir(imageDir)))
  

    numImages = len(imageNames)

    predictions = [None] * numImages
    

    for a in range(numImages):

        test_image = image.load_img(
            imageDir+str(imageNames[a]),
            target_size = (64, 64))

        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = cnn.predict(test_image)

        for predClass in trainingSet.class_indices.keys():
            if result[0][0] == trainingSet.class_indices[predClass]:
                prediction = predClass
        
        predictions[a] = {"prediction":prediction,"fileName": imageNames[a]}
    return predictions
