import csv

import numpy as np

def read_data(file_name):
    """
    Read data from a csv file.

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
    images_as_array = data[:,1:]

    n_img = len(labels)
    label_list = [labels[i] for i in range(n_img)]
    image_list = [images_as_array[i]/255.0 for i in range(n_img)]

    return image_list, label_list






