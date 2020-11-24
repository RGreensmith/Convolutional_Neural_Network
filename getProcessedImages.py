
from keras.preprocessing.image import ImageDataGenerator


def getProcessedImages (isTest = True) :
    """
    1. Gets images from the current working directory, 
    expects the test data and training data to be in folders 
    named test_set and train_set respectively.

    Expected file structure:
    ./
    |--train_set/
    |--test_set/

    2. Rescales the images by 1./255

    3. If training set, augments the images using:
    shear (0.2), zoom (0.2) and horizontal flip

    4. Resizes images (64 x 64)

    5. Sets class mode as binary

    Args:
        isTest (bool, optional): True returns a test dataset, False returns a training dataset. Defaults to True.

    Returns:
        `DirectoryIterator`: dataset ready to be used in a keras Sequential Convolutional Neural Network
    """
    rescale = 1./255
    if isTest :
        datagen = ImageDataGenerator(
            rescale = rescale
        )
   
    else :
        datagen = ImageDataGenerator(
            rescale = rescale,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True
        )

    return datagen.flow_from_directory(
        directory = ('test_set' if isTest else 'train_set'),
        target_size = (64, 64),
        class_mode = 'binary'
    )
