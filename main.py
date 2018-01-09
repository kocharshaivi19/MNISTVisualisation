import tensorflow as tf
from mnist_gan_network import *
from mnist_simple import *
from mnist_visualisation import *
import os
from tensorflow.examples.tutorials.mnist import input_data

BASE_DIR = os.path.basename(os.path.basename(__file__))

if __name__ == '__main__':
    import argparse
    FLAGS = None
    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir', type=str,
                        default=os.path.join(BASE_DIR, 'input_data'),
                        help='Directory for storing input data')

    parser.add_argument('--batch_size', type=int,
                        default=1,
                        help='The size of batch images [32]')

    parser.add_argument('--simple_checkpoint_dir', type=str,
                        default=os.path.join(BASE_DIR, 'simple_checkpoints'),
                        help='Directory to store Checkpoints for the model')

    parser.add_argument('--dcgan_checkpoint_dir', type=str,
                        default=os.path.join(BASE_DIR, 'dcgan_checkpoints'),
                        help='Directory to store Checkpoints for the model')

    parser.add_argument('--simple_savedir', type=str,
                        default=os.path.join(BASE_DIR, 'checkpoints', 'save_simple'),
                        help='Saving generated pictures')

    parser.add_argument('--dcgan_savedir', type=str,
                        default=os.path.join(BASE_DIR, 'dcgan_checkpoints', 'save_dcgan'),
                        help='Directory to save samples')

    parser.add_argument('--vis_savedir', type=str,
                        default=os.path.join(BASE_DIR, 'save_viz'),
                        help='Saving generated pictures')

    parser.add_argument('--is_train', default=True,
                        action='store_true',
                        help="True if training mode")


    FLAGS, unparsed = parser.parse_known_args()

    dataset = input_data.read_data_sets(FLAGS.datadir, one_hot=True)

    if FLAGS.vis_savedir and not tf.gfile.Exists(FLAGS.vis_savedir):
        tf.gfile.MakeDirs(FLAGS.vis_savedir)

    if not tf.gfile.Exists(FLAGS.simple_checkpoint_dir) and not tf.gfile.Exists(FLAGS.dcgan_checkpoint_dir):
        print ("Preparing Training...")
        # Train Simple MNIST model
        snet = MNISTsimple(dataset=dataset, FLAGS=FLAGS)
        snet.trainImageNet()
        print ("Simple Model Training done!")

        # Train DCGAN MNIST model
        dnet = MNISTdcgan(dataset=dataset, FLAGS=FLAGS)
        dnet.mnist_gan()
        print ("DCGAN Model Training done!")

    # create visualization
    net = GradientImage(dataset=dataset, image_size=28, FLAGS=FLAGS)
    print("Loading graphs for Simple and DCGAN...")
    net.load_models()
    print("Starting Gradient-based Image Morphing...")
    net.visualization()
