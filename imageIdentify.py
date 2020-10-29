import numpy as np
from keras.preprocessing import image

def identify (imageName,cnn):

    """[summary]

    Returns:
        [type]: [description]
    """
    
    predictions = [None] * len(imageName)
    # add value names
    for a in range(len(imageName)) :

        test_image = image.load_img(
            'single_prediction/',imageName[a],
            target_size = (64, 64))

        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = cnn.predict(test_image)

        if result[0][0] == 1:
            prediction[a] = training_set.class_indices.keys()
        else :
            prediction[a] = training_set.class_indices.keys()
        
    return = predictions