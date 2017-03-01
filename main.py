import csv
import sys
import random

import data_handling as dh
import autoencoder as ae

# Autoencoder parameters
n_epochs = 20
mini_batch_size = 256
learning_rate = 0.001
standard_deviation = 0.01
initial_bias = 0.01
loss = "SCE" # There are two types of loss: L2 or SCE i.e. Sigmoid Cross Entropy
verbosity_level = False

network_structure_list = [
                          dh.MNIST_WIDTH * dh.MNIST_HEIGHT, # Input layer size
                          32,
                          dh.MNIST_WIDTH * dh.MNIST_HEIGHT # Output layer size
                         ]

# Read input data
file_name = sys.argv[1]
data = dh.read_data(file_name)
random.shuffle(data)

N = len(data)
nvd = int(0.1*N)
print("Number of images: " + str(N))
print("Number of validation images: " + str(nvd))
print("Number of training data: " + str(N - nvd))

training_data = data[nvd:]
validation_data = data[0:nvd]

# Setup the network
ae_one = ae.AutoEncoder(dh.MNIST_WIDTH,
                        dh.MNIST_HEIGHT,
                        network_structure_list,
                        standard_deviation,
                        initial_bias,
                        loss,
                        verbosity_level)

# Train the network.
ae_one.train(training_data,
             validation_data,
             n_epochs,
             mini_batch_size,
             learning_rate)




