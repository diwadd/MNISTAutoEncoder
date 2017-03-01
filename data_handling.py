import csv
import time

import numpy as np
import matplotlib.pyplot as plt

MNIST_WIDTH = 28 # MNIST image dimension - width
MNIST_HEIGHT = 28 # MNIST image dimension = height

def read_data(file_name):
    """
    Read data from a csv file.
    We are using the data sets from Kaggle.
    See:
    - https://www.kaggle.com/c/digit-recognizer/data
    Files:
    - train.csv
    - test.csv

    :param file_name:
    :return data: A numpy array without the first row (labels).
    """
    with open(file_name, "r") as f:
        r = csv.reader(f)
        lr = list(r)
    data = np.array(lr[1:], dtype=np.float32)
    return data


def split_labels_images(data):
    """
    The data is assumed to be a N x M array. The first column is the label.
    The rest of the row is a MNIST image.

    The function splits the data into two lists : label_list and image_list.
    The i-th entry in labels_list contains the label that corresponds
    to the i-th image in image_list.

    :param data:
    :return image_list, label_list:
    """
    labels = data[:,0]
    images_as_array = data[:,1:] / 255.0

    n_img = len(labels)
    label_list = [labels[i] for i in range(n_img)]
    image_list = [images_as_array[i] for i in range(n_img)]

    return image_list, label_list


def print_verbose(text, verbosity_level):
    if verbosity_level == 1:
        print(text)
    else:
        pass


def print_image(autoencoder_input, autoencoder_before_training, autoencoder_output):

    N = len(autoencoder_input)
    rows = 3

    for i in range(N):
        ax = plt.subplot(rows, N, i + 1)
        plt.imshow(autoencoder_input[i].reshape(MNIST_WIDTH, MNIST_HEIGHT), interpolation="nearest")
        plt.gray()
        plt.colorbar()

        ax = plt.subplot(rows, N, i + 1 + N)
        plt.imshow(autoencoder_before_training[i].reshape(MNIST_WIDTH, MNIST_HEIGHT), interpolation="nearest")
        plt.gray()
        plt.colorbar()

        ax = plt.subplot(rows, N, i + 1 + 2*N)
        plt.imshow(autoencoder_output[i].reshape(MNIST_WIDTH, MNIST_HEIGHT), interpolation="nearest")
        plt.gray()
        plt.colorbar()

    plt.show()








