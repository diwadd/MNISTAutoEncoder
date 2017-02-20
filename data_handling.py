import csv

import numpy as np

def read_data(file_name):
    """

    :param file_name:
    :return:
    """
    with open(file_name, "r") as f:
        r = csv.reader(f)
        lr = list(r)
    data = np.array(lr[1:], dtype=np.float32)
    return data


def split_labels_images(data):

    labels = data[:,0]
    images_as_array = data[:,1:]

    n_img = len(labels)
    label_list = [None for i in range(n_img)]
    image_list = [None for i in range(n_img)]

    for i in range(n_img):
        label_list[i] = labels[i]
        image_list[i] = images_as_array[i]/255.0

    return image_list, label_list






