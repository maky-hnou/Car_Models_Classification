"""Main file to run all actions from."""
from test import Test

import numpy as np
from train import TrainModel
from utils import image_to_npy, list_categories

if (__name__ == '__main__'):
    action = input('INFO: Please choose an action:\n'
                   '0: to create json file of categories.\n'
                   '1: to convert image to npy file.\n'
                   '2: to run the training.\n'
                   '3: to test the model.\n'
                   'action: ')
    if (action == '0'):
        print('INFO: Please provide the data path')
        path = input('path to data: ')
        list_categories(path)
    elif (action == '1'):
        print('INFO: Please provide the path to the images and the filename')
        path = input('path to the images: ')
        filename = input('the npy filename: ')
        image_to_npy(filename=filename, path=path, img_size=(64, 64))
    elif (action == '2'):
        print('INFO: Please provide the data path')
        data_path = input('data path: ')
        data = np.load(data_path, allow_pickle=True)
        images = np.array([i[0] for i in data])
        labels = np.array([i[1] for i in data])
        run_training = TrainModel(train_x=images, train_y=labels)
        run_training.train()
    elif (action == '3'):
        print('INFO: Please provide the image to classify and the model path!')
        image_path = input('image path: ')
        model_path = input('modelpath: ')
        run_classification = Test(image_path=image_path, graph_path=model_path)
        category = run_classification.classify()
        print(category)
    else:
        print('ERROR: Wrong choise of action!')
