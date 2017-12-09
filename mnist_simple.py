import tensorflow as tf
import numpy as np
import os

class MNISTsimple():
    def __init__(self, dataset, FLAGS=None):
        self.mnist = dataset
        self.FLAGS = FLAGS

    def weight_variable(self, shape, name=None):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name=None):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(self, x, W, name=None):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

    def max_pool_2x2(self, x, name=None):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def trainImageNet(self):
        # Import data
        tf.set_random_seed(1)
        test_labels = self.mnist.test.labels.nonzero()[1]

        # Create the model
        x = tf.placeholder(tf.float32, [None, 784], name="input_x")
        y_ = tf.placeholder(tf.float32, [None, 10], name="input_y")
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        print("Initialise values - x, y")

        W_conv1 = self.weight_variable([5, 5, 1, 32], name="W_conv1")
        b_conv1 = self.bias_variable([32], name="b_conv1")
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1, name="relu_conv1")
        h_pool1 = self.max_pool_2x2(h_conv1, name="pool_conv1")
        print("Initialise Con1")

        W_conv2 = self.weight_variable([5, 5, 32, 64], name="W_conv2")
        b_conv2 = self.bias_variable([64], name="b_conv2")
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2, name="relu_conv2")
        h_pool2 = self.max_pool_2x2(h_conv2, name="pool_conv2")
        print("Initialise Con2")

        W_fc1 = self.weight_variable([7 * 7 * 64, 1024], name="W_fc1")
        b_fc1 = self.bias_variable([1024], name="b_fc1")

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="relu_h_fc1")
        print("Initialise FC1")

        # Add dropout
        with tf.name_scope("dropout"):
            keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
            print("Initialise Dropout")

        # Add output
        with tf.name_scope("output"):
            W_fc2 = self.weight_variable([1024, 10], name="W_fc2")
            b_fc2 = self.bias_variable([10], name="b_fc2")
            print("Initialise Output")

        y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name="y_conv")
        tf.add_to_collection("y_conv", y_conv)

        # Define loss and optimizer
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        tf.add_to_collection("cross_entropy", cross_entropy)
        print ("Added csentropy")
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # initialise the variables
        init = tf.global_variables_initializer()
        # Create a saver object which will save all the variables
        saver = tf.train.Saver()
        # Train
        rounds = 1
        total_step = 1000

        # Running first session
        print("Starting 1st session...")
        with tf.Session() as sess:
            with tf.device("/gpu:0"):
                # Initialize variables
                sess.run(init)
                if os.path.isdir(self.FLAGS.simple_checkpoint_dir):
                    print ("Model already exists ...")
                else:
                    for step in range(total_step):
                        batch_xs, batch_ys = self.mnist.train.next_batch(100)
                        sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
                        # print(ypred.shape)
                        if step % rounds == 0 or step >= total_step - 1:
                            ypred = sess.run(y_conv, feed_dict={x: self.mnist.test.images, keep_prob: 1})[:, test_labels]
                            ypred = 1.0 / (np.exp(-ypred) + 1.0)
                            print(step)
                            rounds = max(rounds + 5, int(rounds * 1.4))

                    # Test trained model
                    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    print(sess.run(accuracy, feed_dict={x: self.mnist.test.images, y_: self.mnist.test.labels, keep_prob: 1}))
                    # save checkpoint_v1 of the model
                    print("{} Saving checkpoint_v1 of model...".format(step))
                    # check if the folder is present or not to save the model checkpoint_v1
                    checkpoint_name = os.path.join(self.FLAGS.simple_checkpoint_dir, 'imagenet_model.ckpt')
                    saver.save(sess, checkpoint_name, global_step=step)