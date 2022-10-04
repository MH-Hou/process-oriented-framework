import tensorflow as tf2
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow_probability as tfp
import numpy as np


tfd = tfp.distributions

class VAE:
    def __init__(self, dim_z, kl_weight, learning_rate):
        self.dim_x = (28, 28, 1)
        self.dim_z = dim_z
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate

    def encoder_z(self):
        # define prior distribution for the code, which is an isotropic Gaussian
        prior = tfd.Independent(tfd.Normal(loc=tf.zeros(self.dim_z), scale=1.),
                                reinterpreted_batch_ndims=1)
        # build layers argument for tfk.Sequential()
        input_shape = self.dim_x
        layers = [tfkl.InputLayer(input_shape=input_shape)]
        layers.append(tfkl.Conv2D(filters=32, kernel_size=3, strides=(2, 2),
                                  padding='valid', activation='relu'))
        layers.append(tfkl.Conv2D(filters=64, kernel_size=3, strides=(2, 2),
                                  padding='valid', activation='relu'))
        layers.append(tfkl.Flatten())
        # the following two lines set the output to be a probabilistic distribution
        layers.append(tfkl.Dense(tfpl.IndependentNormal.params_size(self.dim_z),
                                 activation=None, name='z_params'))
        layers.append(tfpl.IndependentNormal(self.dim_z,
                                             convert_to_tensor_fn=tfd.Distribution.sample,
                                             activity_regularizer=tfpl.KLDivergenceRegularizer(prior,
                                                                                               weight=self.kl_weight),
                                             name='z_layer'))
        return tfk.Sequential(layers, name='encoder')



