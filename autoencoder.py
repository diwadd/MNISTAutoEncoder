import sys
import time
import random

import tensorflow as tf

import data_handling as dh

TF_LOCAL_DTYPE = tf.float32
print("TF_LOCAL_DTYPE: %s" % (str(TF_LOCAL_DTYPE)))


def weight_variable(shape, standard_deviation, variable_name=None):

    if variable_name == None:
        initial = tf.truncated_normal(shape, 
                                      stddev=standard_deviation, 
                                      dtype=TF_LOCAL_DTYPE)
    else:
        initial = tf.truncated_normal(shape, 
                                      stddev=standard_deviation, 
                                      dtype=TF_LOCAL_DTYPE,
                                      name=variable_name)
    return tf.Variable(initial, dtype=TF_LOCAL_DTYPE)        

    
def bias_variable(shape, initial_bias, variable_name=None):

    if variable_name == None:
        initial = tf.constant(initial_bias, 
                              shape=shape, 
                              dtype=TF_LOCAL_DTYPE)
    else:
        initial = tf.constant(initial_bias, 
                             shape=shape, 
                             dtype=TF_LOCAL_DTYPE,
                             name=variable_name)
    return tf.Variable(initial, dtype=TF_LOCAL_DTYPE)


class AutoEncoder:

    def ae_network(self,
                   network_structure_list,
                   standard_deviation,
                   initial_bias):
        """
        Builds the autoencoder network.
        The structue of the network is defined in the network_structure_list.

        The network_structure_list should have the following structue:

        network_structure_list = [
                                  input_size,
                                  first_hidden_layer_size,
                                  second_hidden_layer_size,
                                  ...
                                  output_size
                                  ]

        :param network_structure_list:
        :param standard_deviation:
        :param initial_bias:
        :return:
        """

        self.network_input = tf.placeholder(TF_LOCAL_DTYPE,
                                            shape=[None, network_structure_list[0]],
                                            name="network_input")

        current_layer = self.network_input
        print("Input layer shape %15s" % (str(current_layer.get_shape())))

        print("len(network_structure_list): " + str(len(network_structure_list)))
        for i in range(1, len(network_structure_list)-1):
            print("i: " + str(i))
            w_connected = weight_variable([network_structure_list[i-1], network_structure_list[i]], 
                                          standard_deviation,
                                          variable_name="w_" + str(i) + "_connected")

            b_connected = bias_variable([network_structure_list[i]], 
                                        initial_bias,
                                        variable_name="b_" + str(i) + "_connected")

            mm = tf.matmul(current_layer, w_connected)
            connected_layer = tf.nn.relu(mm + b_connected)
            current_layer = connected_layer

            dh.print_verbose("w_connected %5s shape %15s" % (str(i), str(w_connected.get_shape())), self.verbosity_level)
            dh.print_verbose("b_connected %5s shape %15s" % (str(i), str(b_connected.get_shape())), self.verbosity_level)
            dh.print_verbose("mm %5s shape %15s" % (str(i), str(mm.get_shape())), self.verbosity_level)
            dh.print_verbose("Layer %5s shape %15s" % (str(i), str(connected_layer.get_shape())), self.verbosity_level)

        w_connected = weight_variable([network_structure_list[-2], network_structure_list[-1]], standard_deviation)
        b_connected = bias_variable([network_structure_list[-1]], initial_bias)


        if self.loss == "L2":
            self.network_output = tf.sigmoid(tf.matmul(current_layer, w_connected) + b_connected)
            self.network_output_for_prediction = tf.sigmoid(tf.matmul(current_layer, w_connected) + b_connected)

            self.network_expected_output = tf.placeholder(TF_LOCAL_DTYPE,
                                                          shape=[None, network_structure_list[-1]],
                                                          name="network_expected_output")
        elif self.loss == "SCE": # SCE - Sigmoid Cross Entropy
            self.network_output = tf.matmul(current_layer, w_connected) + b_connected
            self.network_output_for_prediction = tf.sigmoid(tf.matmul(current_layer, w_connected) + b_connected)

            self.network_expected_output = tf.placeholder(TF_LOCAL_DTYPE,
                                                          shape=[None, network_structure_list[-1]],
                                                          name="network_expected_output")
        else:
            pass

        print("Network output shape %15s" % (str(self.network_output.get_shape())))
        print("Network expected output shape %15s" % (str(self.network_expected_output.get_shape())))



    def __init__(self,
                 width,
                 height,
                 network_structure_list,
                 standard_deviation,
                 initial_bias,
                 loss,
                 verbosity_level=False):

        self.width = width
        self.height = height
        self.network_structure_list = network_structure_list
        self.standard_deviation = standard_deviation
        self.initial_bias = initial_bias
        self.loss = loss
        self.verbosity_level = verbosity_level

        self.parameter_dict = None

        self.ae_network(self.network_structure_list,
                        self.standard_deviation,
                        self.initial_bias)



    def setup_loss(self):
        """
        Set the loss function.

        L2 is the least squares loss typically used in regression tasks.
        SCE stands for Sigmoid Cross Entropy.

        :param mini_batch_size:
        :return:
        """

        if self.loss == "L2":
            t = tf.subtract(self.network_output, self.network_expected_output)
            self.C = tf.nn.l2_loss(t)
        elif self.loss == "SCE":
            self.C = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.network_expected_output,
                                                                            logits=self.network_output))
        else:
            pass

    def setup_minimize(self, learning_rate):
        """
        The the minimization algorithm.
        Usa a decaying learning rate.

        :param learning_rate:
        :param decay_steps:
        :param decay_rate:
        :return:
        """

        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.C)


    def setup_session(self):

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    def set_parameter_dict_for_train(self, images):

        self.parameter_dict = {self.network_input: images,
                               self.network_expected_output: images}



    def train(self,
              data,
              n_epochs,
              mini_batch_size,
              learning_rate):

        self.setup_loss()
        self.setup_minimize(learning_rate)
        self.setup_session()

        n_batches_per_epoch = round(len(data) / mini_batch_size)

        print("Number of mini batches: " + str(n_batches_per_epoch))

        for epoch in range(n_epochs):
            ptr = 0

            train_loss = 0.0
            for batch in range(n_batches_per_epoch):
                # start = time.time()
                mini_batch = data[ptr:ptr + mini_batch_size]
                ptr = ptr + mini_batch_size


                # This slows the code down. For such a simple dataset
                # this should be preapplied to the data before the train
                # method is applied.
                images, _ = dh.split_labels_images(mini_batch)
                


                self.set_parameter_dict_for_train(images)

                (self.train_step).run(session=self.sess, feed_dict=self.parameter_dict)

                # stop = time.time()
                # print("Mini batch time: " + str(stop - start))

            images, _ = dh.split_labels_images(data)
            self.set_parameter_dict_for_train(images)
            c_val_train = (self.C).eval(session=self.sess, feed_dict=self.parameter_dict)


            print("(epoch %10s) C value: %10s" % (str(epoch), str(c_val_train)))

        min_index = random.randint(0, len(images) - 8 - 1)
        max_index = min_index + 8        
        self.set_parameter_dict_for_train(images[min_index:max_index])
        image_from_ae = (self.network_output_for_prediction).eval(session=self.sess, feed_dict=self.parameter_dict)

        
        dh.print_image(images[min_index:max_index], image_from_ae)





