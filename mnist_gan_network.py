from __future__ import print_function
import math
import numpy as np
import os
import scipy.misc
import tensorflow as tf

slim = tf.contrib.slim
tf.set_random_seed(1)
np.random.seed(1)
tf.logging.set_verbosity(tf.logging.INFO)

class MNISTdcgan():
    def __init__(self, dataset,
                 batch_size=16,
                 FLAGS=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.FLAGS = FLAGS

    def merge(self, images, size):
        '''
        Tiles the images in a batch into single image
        :param images: Batch of images
        :param size: size of the patch in final image
        :return: patched image
        '''
        print ("image", images.shape)
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx // size[1])
            print ("i : {0}, j : {1}".format(i, j))
            img[j * h:j * h + h, i * w:i * w + w] = image
        return img

    def lrelu(self, x, leak=0.2, name="lrelu"):
        '''
        Performs Leaky Relu operation
        :param x: input tensor
        :param leak: float value to perform leaky operation, ranging from 0 to 1
        :param name: name of the tensor
        :return: resultant tensor
        '''
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)

    def generator(self, z, reuse=True):
        '''
        Builds the Generative model for MNIST Data
        :param z: input z tensor
        :param reuse: boolean value to reuse tensors
        :return: model
        '''
        init_width = 7
        filters = (256, 128, 64, 1)
        kernel_size = 4
        with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                            reuse=reuse,
                            normalizer_fn=slim.batch_norm):
            with tf.variable_scope("gen"):
                net = slim.fully_connected(
                    z, init_width ** 2 * filters[0], scope='fc1')
                net = tf.reshape(net, [-1, init_width, init_width, filters[0]])
                net = slim.conv2d_transpose(
                    net, filters[1],
                    kernel_size=kernel_size,
                    stride=2,
                    scope='deconv1')
                net = slim.conv2d_transpose(
                    net, filters[2],
                    kernel_size=kernel_size,
                    stride=1,
                    scope='deconv2')
                net = slim.conv2d_transpose(
                    net,
                    filters[3],
                    kernel_size=kernel_size,
                    stride=2,
                    activation_fn=tf.nn.tanh,
                    scope='deconv3')
                tf.summary.histogram('gen/out', net)
                tf.summary.image("gen", net, max_outputs=8)
        return net


    def discriminator(self, x, reuse):
        '''
        Builds the discriminator model on MNIST Data
        :param x: input image tensor (either from Generator or original)
        :param reuse: boolean value to reuse the tensors
        :return: model
        '''
        filters = (32, 64, )
        kernels = (4, 4)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            reuse=reuse,
                            activation_fn=self.lrelu):
            with tf.variable_scope("discr"):
                net = tf.reshape(x, [-1, 28, 28, 1])
                net = slim.conv2d(net, filters[0], kernels[0], stride=2, normalizer_fn=slim.batch_norm, scope='conv1')
                net = slim.conv2d(net, filters[1], kernels[1], stride=2, normalizer_fn=slim.batch_norm, scope='conv2')
                net = slim.flatten(net,)
                net = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid, scope='out')
        return net


    def mnist_gan(self):
        '''
        Model Generation and Training
        :param dataset: input dataset
        '''
        z_dim = 100
        x = tf.placeholder(tf.float32, shape=[None, 784], name='X')
        d_model = self.discriminator(x, reuse=False)

        z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
        g_model = self.generator(z, reuse=False)
        dg_model = self.discriminator(g_model, reuse=True)

        tf.add_to_collection("d_model", d_model)
        tf.add_to_collection("dg_model", dg_model)
        tf.add_to_collection('g_model', g_model)

        # Optimizers
        t_vars = tf.trainable_variables()
        global_step = tf.Variable(0, name='global_step', trainable=False)
        d_loss = -tf.reduce_mean(tf.log(d_model) + tf.log(1. - dg_model), name='d_loss')
        tf.summary.scalar('d_loss', d_loss)
        d_trainer = tf.train.AdamOptimizer(.0002, beta1=.5, name='d_adam').minimize(
            d_loss,
            global_step=global_step,
            var_list=[v for v in t_vars if 'discr/' in v.name],
            name='d_min')

        g_loss = -tf.reduce_mean(tf.log(dg_model), name='g_loss')
        tf.summary.scalar('g_loss', g_loss)
        g_trainer = tf.train.AdamOptimizer(.0002, beta1=.5, name='g_adam').minimize(
            g_loss, var_list=[v for v in t_vars if 'gen/' in v.name],
            name='g_adam')
        init = tf.global_variables_initializer()
        # Session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            with tf.device('/gpu:0'):
                sess.run(init)

                # Savers
                if not tf.gfile.Exists(self.FLAGS.dcgan_checkpoint_dir):
                    tf.gfile.MakeDirs(self.FLAGS.dcgan_checkpoint_dir)

                saver = tf.train.Saver(max_to_keep=20)
                checkpoint = tf.train.latest_checkpoint(self.FLAGS.dcgan_checkpoint_dir)
                if checkpoint and not self.FLAGS.debug:
                    print('Restoring from', checkpoint)
                    saver.restore(sess, checkpoint)
                summary = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(self.FLAGS.dcgan_checkpoint_dir, sess.graph)

                # Training loop
                for step in range(int(1e6)):
                    z_batch = np.random.uniform(-1, 1, [self.batch_size, z_dim]).astype(np.float32)
                    images, labels = self.dataset.train.next_batch(self.batch_size)

                    # Update discriminator
                    _, d_loss_val = sess.run([d_trainer, d_loss], feed_dict={x: images, z: z_batch})
                    # Update generator twice
                    sess.run(g_trainer, feed_dict={z: z_batch})
                    _, g_loss_val = sess.run([g_trainer, g_loss], feed_dict={z: z_batch})

                    # Log details
                    print("Gen Loss: ", g_loss_val, " Disc loss: ", d_loss_val)
                    print (z_batch.shape)
                    summary_str = sess.run(summary, feed_dict={x: images, z: z_batch})
                    summary_writer.add_summary(summary_str, global_step.eval())

                    # Early stopping
                    if np.isnan(g_loss_val) or np.isinf(g_loss_val) \
                            or np.isnan(d_loss_val) or d_loss_val == -0.0:
                        print('Early stopping')
                        break

                    if step % 100 == 0:
                        # Save samples
                        if self.FLAGS.dcgan_savedir:
                            samples = 64
                            z2 = np.random.uniform(-1.0, 1.0, size=[samples, z_dim]).astype(np.float32)
                            images = sess.run(g_model, feed_dict={z: z2})
                            images = np.reshape(images, [samples, 28, 28])
                            images = (images + 1.) / 2.
                            scipy.misc.imsave(self.FLAGS.dcgan_savedir + '/sample'+str(step)+'.png',
                                              self.merge(images, [int(math.sqrt(samples))] * 2))

                            # save model
                            checkpoint_file = os.path.join(self.FLAGS.dcgan_checkpoint_dir, 'checkpoint')
                            saver.save(sess, checkpoint_file, global_step=global_step)
                return