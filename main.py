import csv
import sys

import data_handling as dh
import autoencoder as ae

# Input data and autoencoder parameters
file_name = sys.argv[1]
width = 28
height = 28
n_epochs = 100000
mini_batch_size = 1000
initial_learning_rate = 0.5
decay_steps = 500
decay_rate = 0.9
standard_deviation = 0.01
initial_bias = 0.01

network_structure_list = [
                          width * height,
                          128,
                          32,
                          128,
                          width * height
                         ]

# Read input data
data = dh.read_data(file_name)
image_list, label_list = dh.split_labels_images(data)

print("Number of images: " + str(len(data)))

# Setup the network
ae_one = ae.AutoEncoder(width,
                        height,
                        network_structure_list,
                        standard_deviation,
                        initial_bias)

# Train the network.
ae_one.train(data,
             n_epochs,
             mini_batch_size,
             initial_learning_rate,
             decay_steps,
             decay_rate)




