import csv
import sys

import data_handling as dh
import autoencoder as ae

# Autoencoder parameters
n_epochs = 60
mini_batch_size = 256
learning_rate = 0.001
standard_deviation = 0.01
initial_bias = 0.01
loss = "SCE" # There are two types of loss: L2 or SCE i.e. Sigmoid Cross Entropy
verbosity_level = False

network_structure_list = [
                          dh.MNIST_WIDTH * dh.MNIST_HEIGHT, # Input layer size
                          48,
                          16,
                          48,
                          dh.MNIST_WIDTH * dh.MNIST_HEIGHT # Output layer size
                         ]

# Read input data
file_name = sys.argv[1]
data = dh.read_data(file_name)
image_list, label_list = dh.split_labels_images(data)

print("Number of images: " + str(len(data)))

# Setup the network
ae_one = ae.AutoEncoder(dh.MNIST_WIDTH,
                        dh.MNIST_HEIGHT,
                        network_structure_list,
                        standard_deviation,
                        initial_bias,
                        loss,
                        verbosity_level)

# Train the network.
ae_one.train(data,
             n_epochs,
             mini_batch_size,
             learning_rate)




