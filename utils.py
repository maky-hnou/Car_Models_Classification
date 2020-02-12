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
