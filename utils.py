"""This helper script loads the ASL-Alphabet dataset,
which can be downloaded from Kaggle:

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

The script comes originally from a Datacamp project,
but it was modified to work with the Kaggle version of the dataset.

"""

import random
import numpy as np
from keras.utils import np_utils, to_categorical
from tensorflow.keras.preprocessing import image
from os import listdir
from os.path import isdir, join

def load_data(container_path='data/asl_alphabet_train/asl_alphabet_train',
              #folders=['A', 'B', 'C'],
              folders=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                       'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                       'S', 'T', 'U', 'W', 'X', 'Y', 'Z'],
              size=None, # 2000
              test_split=0.2,
              seed=0,
              transfer_learning=False):
    """
    Loads sign language dataset and
    returns it as
    (x_train, y_train), (x_test, y_test)
    """
    
    filenames, labels = [], []

    for label, folder in enumerate(folders):
        folder_path = join(container_path, folder)
        images = [join(folder_path, d)
                     for d in sorted(listdir(folder_path))]
        labels.extend(len(images) * [label])
        filenames.extend(images)

    random.seed(seed)
    data = list(zip(filenames, labels))
    random.shuffle(data)
    if size:
        # Size not None, we pick the number of samples specified
        if size > len(filenames):
            size = len(filenames)
        data = data[:size]
    filenames, labels = zip(*data)
    
    # Get the images
    x = paths_to_tensor(filenames, transfer_learning=transfer_learning).astype('float32')/255
    # Store the one-hot targets
    y = np.array(labels)

    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])

    return (x_train, y_train), (x_test, y_test)
    #return (x, y)

def path_to_tensor(img_path, size, transfer_learning=False):
    """Load image from path and return it as tensor."""
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(size, size))
    # convert PIL.Image.Image type to 3D tensor
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor 
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, size=50, transfer_learning=False):
    """Load images from a list of paths and return them as tensors."""
    list_of_tensors = [path_to_tensor(img_path, size, transfer_learning) for img_path in img_paths]
    return np.vstack(list_of_tensors)
