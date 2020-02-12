"""Utils functions."""

import glob
import math
import os
import random
import re

import cv2
import numpy as np
from natsort import natsorted, ns
from tqdm import tqdm


def image_to_npy(filename, path, img_size, json_file_path):
    """Create numpy array from an input image.

    Parameters
    ----------
    filename : str
        The name of the array of images.
    path : str
        The path of the directory where the data is.
    img_size : tuple
        The new size of the image.
    json_file_path : str
        The path of the json file.

    Returns
    -------
    None

    """
    data = []
    for img in tqdm(natsorted(glob.glob(path + '/**/*'), alg=ns.IGNORECASE)):
        label = label_images(img, path, json_file_path)
        img = cv2.imread(img, 1)
        img = cv2.resize(img, img_size)
        data.append([np.array(img), label])
    random.shuffle(data)
    np.save('{}_data_gray.npy'.format(filename), data)


def label_images(filename, path, json_file_path):
    """Return the label of a given image existing a given directory.

    Parameters
    ----------
    filename : str
        The filename path.
    path : str
        The directory path.
    json_file_path : str
        The path of the json file.

    Returns
    -------
    list
        The label of the given image.

    """
    categories_dict = get_categories_names(json_file_path)
    classes_number = len(glob.glob(path + '/*'))
    class_labels = np.identity(classes_number, dtype=int)
    class_label = list(categories_dict.keys())[
        list(categories_dict.values()).index(filename.split('/')[-2])]
    return class_labels[int(class_label) - 1]


def rename_files(path):
    """Rename files in subdirectores of a given parent directory.

    Parameters
    ----------
    path : str
        The parent directory path.

    Returns
    -------
    None

    """
    for folder in natsorted(glob.glob(path + '/*'), alg=ns.IGNORECASE):
        i = 1
        for img in natsorted(glob.glob(folder + '/*'), alg=ns.IGNORECASE):
            path_parts = img.split('/')
            class_folder_path = path_parts[:-1]
            class_number = path_parts[-2]
            os.rename(img, '/'.join(class_folder_path)
                      + '/{}_{}.jpg'.format(class_number, i))
            i += 1
