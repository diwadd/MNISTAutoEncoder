import sys
import time

import tensorflow as tf

import data_handling as dh

TF_LOCAL_DTYPE = tf.float32
print("TF_LOCAL_DTYPE: %s" % (str(TF_LOCAL_DTYPE)))

def weight_variable(shape, standard_deviation):
  initial = tf.truncated_normal(shape, stddev=standard_deviation, dtype=TF_LOCAL_DTYPE)
  return tf.Variable(initial, dtype=TF_LOCAL_DTYPE)

def bias_variable(shape, initial_bias):
  initial = tf.constant(initial_bias, shape=shape, dtype=TF_LOCAL_DTYPE)
  return tf.Variable(initial, dtype=TF_LOCAL_DTYPE)


class AutoEncoder:

    def ae_network(self,
                   network_structure_list,
                   standard_deviation,
                   initial_bias):

        self.network_input = tf.placeholder(TF_LOCAL_DTYPE,
                                            shape=[None, network_structure_list[0][0]],
                                            name="network_input")

        current_layer = self.network_input
        print("Input layer shape %15s" % (str(current_layer.get_shape())))


        print("len(network_structure_list): " + str(len(network_structure_list)))
        for i in range(1, len(network_structure_list)-1):
            print("i: " + str(i))
            w_connected = weight_variable(network_structure_list[i], standard_deviation)
            print("w_connected %5s shape %15s" % (str(i), str(w_connected.get_shape())))

            b_connected = bias_variable([network_structure_list[i][1]], initial_bias)
            print("b_connected %5s shape %15s" % (str(i), str(b_connected.get_shape())))

            mm = tf.matmul(current_layer, w_connected)
            print("mm %5s shape %15s" % (str(i), str(mm.get_shape())))
            connected_layer = tf.nn.relu(mm + b_connected)
            print("Layer %5s shape %15s" % (str(i), str(connected_layer.get_shape())))
            current_layer = connected_layer

        w_connected = weight_variable(network_structure_list[-1], standard_deviation)
        b_connected = bias_variable([network_structure_list[-1][1]], initial_bias)

        self.network_output = tf.matmul(current_layer, w_connected) + b_connected
        print("Network output shape %15s" % (str(self.network_output.get_shape())))

        self.network_expected_output = tf.placeholder(TF_LOCAL_DTYPE,
                                                      shape=[None, network_structure_list[-1][1]],
                                                      name="network_expected_output")
        print("Network expected output shape %15s" % (str(self.network_expected_output.get_shape())))



    def __init__(self,
                 width,
                 height,
                 network_structure_list,
                 standard_deviation,
                 initial_bias):



        self.width = width
        self.height = height
        self.network_structure_list = network_structure_list
        self.standard_deviation = standard_deviation
        self.initial_bias = initial_bias

        self.parameter_dict = None

        self.ae_network(self.network_structure_list,
                        self.standard_deviation,
                        self.initial_bias)



    def setup_loss(self, mini_batch_size):

        self.mini_batch_size = mini_batch_size

        s = tf.subtract(self.network_output, self.network_expected_output)
        self.C = tf.reduce_sum(tf.multiply(s, s))
        m = tf.constant(2.0 * mini_batch_size, dtype=tf.float32)
        self.C = tf.divide(self.C, m)


    def setup_minimize(self, initial_learning_rate, decay_steps, decay_rate):

        self.global_step = tf.Variable(0, trainable=False)
        self.decaying_learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                                            self.global_step,
                                                            decay_steps,
                                                            decay_rate,
                                                            staircase=True)

        self.train_step = tf.train.AdamOptimizer(self.decaying_learning_rate).minimize(self.C, global_step=self.global_step)

    def setup_session(self):

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)



    def set_parameter_dict_for_train(self, images):

        self.parameter_dict = {self.network_input: images,
                               self.network_expected_output: images}



    def train(self,
              x_train,
              n_epochs,
              mini_batch_size,
              initial_learning_rate,
              decay_steps,
              decay_rate):

        self.setup_loss(mini_batch_size)
        self.setup_minimize(initial_learning_rate, decay_steps, decay_rate)
        self.setup_session()

        n_batches_per_epoch = round(len(x_train) / mini_batch_size)

        print("Number of mini batches: " + str(n_batches_per_epoch))
        print("decaying_learning_rate: " + str(self.decaying_learning_rate.eval(session=self.sess)))
        for epoch in range(n_epochs):
            ptr = 0

            train_loss = 0.0
            for batch in range(n_batches_per_epoch):
                #start = time.time()
                mini_batch = x_train[ptr:ptr + mini_batch_size]
                ptr = ptr + mini_batch_size


                images, _ = dh.split_labels_images(mini_batch)


                self.set_parameter_dict_for_train(images)

                (self.train_step).run(session=self.sess, feed_dict=self.parameter_dict)

                c_val_train = (self.C).eval(session=self.sess, feed_dict=self.parameter_dict)
                #print("(in batch loop, %10s) c_val_train value: %10s" % (str(batch), str(c_val_train)))
                #train_loss = train_loss + (2.0*len(mini_batch))*c_val_train

                #stop = time.time()
                #print("Mini batch time: " + str(stop - start))

            images, _ = dh.split_labels_images(x_train)
            self.set_parameter_dict_for_train(images)
            c_val_train = (self.C).eval(session=self.sess, feed_dict=self.parameter_dict)

            dlr = self.decaying_learning_rate.eval(session=self.sess)
            gs = self.global_step.eval(session=self.sess)

            print("(epoch %10s) C value: %10s learning rate %10s, global step %10s" % (str(epoch),
                                                                                       str(c_val_train),
                                                                                       str(dlr),
                                                                                       str(gs)))





