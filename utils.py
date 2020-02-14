"""Utils functions."""
import glob
import random
import re

import cv2
import numpy as np
from natsort import natsorted, ns
from tqdm import tqdm


def image_to_npy(filename, path, img_size):
    """Convert images to .npy file.

    Parameters
    ----------
    filename : str
        The name of the .npy file to be created.
    path : str
        The path of the images.
    img_size : tuple
        The new size of the image..

    Returns
    -------
    None

    """
    data = []
    for img in tqdm(natsorted(glob.glob(path + '/*/*'), alg=ns.IGNORECASE)):
        image = cv2.imread(img)
        image = cv2.resize(image, img_size)
        label = img.split('/')[-2]
        label = re.sub(r' |\-', '_', label)
        data.append([np.array(image), label])
    random.shuffle(data)
    np.save('{}_data.npy'.format(filename), data)
