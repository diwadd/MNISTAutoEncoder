import csv

import data_handling as dh
import autoencoder as ae


file_name = "train.csv"
width = 28
height = 28

data = dh.read_data(file_name)
image_list, label_list = dh.split_labels_images(data)

print("Number of images: " + str(len(data)))

network_structure_list = [
                          [width * height],
                          [width * height, 100],
                          [100, 10],
                          [10, 100],
                          [100, width * height]
                         ]

standard_deviation = 0.01
initial_bias = 0.01

for i in range(10):
    print(label_list[i], image_list[i].shape)


ae_one = ae.AutoEncoder(width,
                        height,
                        network_structure_list,
                        standard_deviation,
                        initial_bias)


x_train = data
n_epochs = 100000
mini_batch_size = 1000
initial_learning_rate = 10.0
decay_steps = 2000
decay_rate = 0.9


ae_one.train(x_train,
             n_epochs,
             mini_batch_size,
             initial_learning_rate,
             decay_steps,
             decay_rate)




