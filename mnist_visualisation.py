# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.ndimage.filters import gaussian_filter
import os
import scipy
import pdb
import numpy as np
import tensorflow as tf

class GradientImage(object):
    def __init__(self, dataset, image_size, FLAGS=None,
                 weight_decay=0.05,
                 learningrate=0.0007,
                 gaussian_Blur_span=0.3,
                 batch_size=1,
                 percent=49):
        self.dataset = dataset
        self.image_size = image_size
        self.FLAGS = FLAGS
        self.weight_decay = weight_decay
        self.learningrate = learningrate
        self.gaussian_Blur_span = gaussian_Blur_span
        self.batch_size = batch_size
        self.percent = percent

    def blur_regularization(self, img):
        """
        Performs the Gaussian Blur on the image
        :param img: input image
        :param grads:
        :param size:
        :return: image
        """
        return gaussian_filter(img.reshape((self.image_size, self.image_size)),
                               sigma=self.gaussian_Blur_span).reshape((1, self.image_size ** 2))

    def clip_weak_pixel_regularization(self, img):
        """
        Clipping the weak parts of the images by putting the value of the pixel as zero and reviewing the effect of
        pixel on the gradient
        :param img:
        :param grads:
        :param percentile:
        :return:
        """
        clipped = img
        threshold = np.percentile(np.abs(img), self.percent)
        clipped[np.where(np.abs(img) < threshold)] = 0
        return clipped

    def normalize(self, x):
        '''
        Utility function to normalize a tensor by its L2 norm
        :param x:
        :return: Normalised Numpy Image array
        '''
        return x / (np.sqrt(np.mean(np.square(x))) + 1e-5)

    def write_img_display(self, image, filename, pause=0.016):
        '''
        Print and save the image using Matplotlib
        :param image:
        :param filename:
        :param pause:
        :return:
        '''
        image = np.clip(np.reshape(image, (self.image_size, self.image_size)) * 255, 0, 255)
        print (image.shape)
        # plt.title("image")
        # im = plt.imshow(image)
        # cb = plt.colorbar(im)
        # plt.draw()
        # plt.pause(pause)
        # cb.remove()
        scipy.misc.imsave(filename + '.png', image)

    def load_models(self):
        checkpoint_dcgan = tf.train.latest_checkpoint(self.FLAGS.dcgan_checkpoint_dir)
        checkpoint = tf.train.latest_checkpoint(self.FLAGS.simple_checkpoint_dir)
        self.adv_graph = tf.Graph()
        self.model_graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.adv_sess = tf.Session(graph=self.adv_graph, config=config)
        self.sess = tf.Session(graph=self.model_graph, config=config)
        with self.sess.as_default():
            with self.model_graph.as_default():
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint))
                saver.restore(self.sess, checkpoint)
                print("Simple Model Restored ...")
                print(tf.global_variables())
                self.y_conv = tf.get_collection("y_conv")[0]
                print("y_conv", self.y_conv)
                self.x = self.model_graph.get_tensor_by_name("input_x:0")
                print("x_shape: ", self.x)
                self.y = self.model_graph.get_tensor_by_name("input_y:0")
                print("y_shape: ", self.y)
                self.keep_prob = self.model_graph.get_tensor_by_name("dropout/keep_prob:0")
                self.softmax = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_conv)
                print("softmax: ", self.softmax)
                self.loss = tf.reduce_mean(self.softmax)
                self.var_grad = tf.gradients(self.loss, [self.x])
                print(self.loss)
                print(self.var_grad)

        with self.adv_sess.as_default():
            with self.adv_graph.as_default():
                saver_dcgan = tf.train.import_meta_graph("{}.meta".format(checkpoint_dcgan))
                saver_dcgan.restore(self.adv_sess, checkpoint_dcgan)
                print("DCGAN Model Restored ...")
                print(tf.global_variables())
                self.x_dcgan = self.adv_graph.get_tensor_by_name('X:0')
                print("x_dcgan: ", self.x_dcgan)
                self.y_conv_dcgan = self.adv_graph.get_collection('d_model')[0]
                print("y_conv_dcgan", self.y_conv_dcgan)
                self.loss_dcgan = -tf.reduce_mean(tf.log(self.y_conv_dcgan))
                print("loss_dcgan: ", self.loss_dcgan)
                self.var_grad_dcgan = tf.gradients(self.loss_dcgan, [self.x_dcgan])
                print("var_grad_dcgan: ", self.var_grad_dcgan)

    def gradientMorphing(self, x_start, y_desired, sub_dir_name):
        num_file = 0
        for i in range(5000):
            with self.sess.as_default():
                l = self.sess.run(self.loss, feed_dict={self.x: x_start, self.y: y_desired, self.keep_prob: 1})
                x_grad = self.sess.run(self.var_grad[0],
                                       feed_dict={self.x: x_start, self.y: y_desired, self.keep_prob: 1})
            with self.adv_sess.as_default():
                ldc = self.adv_sess.run(self.loss_dcgan, feed_dict={self.x_dcgan: x_start})
                x_dcgan_grad = self.adv_sess.run(self.var_grad_dcgan[0], feed_dict={self.x_dcgan: x_start})
            print("count: {0}, l: {1}, ldcgan: {2}".format(i, l, ldc))
            x_start -= (np.add(x_grad[0], x_dcgan_grad[0]) * self.learningrate +
                        self.weight_decay * self.learningrate * x_start)
            print ("x_start shape: ", x_start.shape)
            for reg in [self.clip_weak_pixel_regularization, self.blur_regularization]:
                x_start = reg(x_start)
            if i % 100 == 0:
                if not tf.gfile.Exists(os.path.join(sub_dir_name)):
                    tf.gfile.MakeDirs(os.path.join(sub_dir_name))

                self.write_img_display(x_start[0], filename=os.path.join(sub_dir_name, str(num_file)))
                num_file += 1

    def createGridView(self, path, from_label):
        frame = np.array([10 * self.image_size, 10 * self.image_size], dtype=np.float32)
        lt = [i for i in range(0, 10)]
        epoch_list = [0, 100, 200, 300]
        savedir = os.path.join(path, "mnist_results")
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        for ep in epoch_list:
            for to_i in lt:
                if to_i == from_label:
                    continue
                else:
                    img = scipy.misc.imread(os.path.join(path, "testing_" + str(from_label) + "_" + str(to_i), str(ep) + ".png"))
                    frame[from_label * self.image_size: from_label * self.image_size + self.image_size,
                            to_i * self.image_size: to_i * self.image_size + self.image_size] = img
            scipy.misc.imsave(os.path.join(savedir, str(ep) + '.png'), frame)

    def visualization(self, random=False):
        digit_list = [i for i in range(0, 10)]
        rix = 0
        idx = []
        for fr in digit_list:
            while rix < len(self.dataset.test.labels):
                if np.nonzero(self.dataset.test.labels[rix])[0] == fr:
                    idx.append(rix)
                    break
                rix += 1

        # generated images from specified digit vector
        for from_label, index in enumerate(idx):
            x_start = self.dataset.test.images[index]
            x_start = np.reshape(x_start, [1, 784])
            for des in digit_list[:from_label] + digit_list[from_label+1:]:
                y_desired = np.zeros((1, 10), dtype=np.float32)
                y_desired[0, des] = 1
                print("Transformtion From Label {0} to Desired Label {1}".format(from_label, des))
                self.gradientMorphing(x_start, y_desired,
                                      sub_dir_name=os.path.join(self.FLAGS.vis_savedir,
                                                    'testing_' + str(from_label) + '_' + str(des)))
            self.createGridView(path=self.FLAGS.vis_savedir, from_label=from_label)

        if random:
            # generated images using random input vector
            for desired_label in digit_list:
                x_start = np.random.normal(0, 0.01, (1, 784))
                y_desired = np.zeros((1, 10), dtype=np.float32)
                y_desired[0, desired_label] = 1
                print("Transformation From Random to Desired Label {0}".format(desired_label))
                self.gradientMorphing(x_start=x_start, y_desired=y_desired,
                                      sub_dir_name='random_' + str(desired_label))
            self.createGridView(path=self.FLAGS.vis_savedir)