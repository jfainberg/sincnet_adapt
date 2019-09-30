import keras
import numpy as np
import math
import os
from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import Layer
from keras.engine.base_layer import InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces
import tensorflow as tf

class SincConv(Layer):
    '''
    Slightly messy version of SincConv, but matches PyTorch implementation.
    '''
    
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    def __init__(self, filters, kernel_size, stride=1, padding='valid',
                 dilation_rate=1, sample_rate=16000, min_low_hz=50.,
                 min_band_hz=50., activation=None, data_format='channels_last',
                 initializer = 'mel', **kwargs):
        super(SincConv, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activations.get(activation)

        self.initializer = initializer
        
        self.stride = stride
        self.padding = padding
        self.data_format = data_format  # (batch, steps, channels)
        self.dilation_rate = dilation_rate

    
        # Odd, symmetric filters
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1
            
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        

        
    def build(self, input_shape):
        # Initialize filterbanks to be equally spaced by Mel scale
        low_hz = 30.0  # must be float
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.filters + 1)
        hz = self.to_hz(mel) / self.sample_rate
        
        # Could move the above into their own init functions instead
        # of using lambda below
        if self.initializer == 'flat':
            low_hz_init = lambda x: np.ones((len(hz[:-1]), 1)) * low_hz / self.sample_rate
            band_hz_init = lambda x: np.ones((len(hz[:-1]), 1)) * 50. / self.sample_rate
        elif self.initializer == 'uniform':
            hz = np.sort(np.random.uniform(low_hz, high_hz, self.filters + 1)) / self.sample_rate
            low_hz_init = lambda x: np.expand_dims(hz[:-1], 1)
            band_hz_init = lambda x: np.expand_dims(np.diff(hz),1)
        elif self.initializer == 'flat_uniform':
            delta = 3. # Hz
            low_hz = np.sort(np.random.uniform(low_hz-delta, low_hz+delta, self.filters)) / self.sample_rate
            band_hz= np.sort(np.random.uniform(50.-delta, 50.+delta, self.filters)) / self.sample_rate
            low_hz_init = lambda x: np.expand_dims(low_hz, 1)
            band_hz_init = lambda x: np.expand_dims(band_hz, 1)
        else: # mel
            low_hz_init = lambda x: np.expand_dims(hz[:-1], 1)
            band_hz_init = lambda x: np.expand_dims(np.diff(hz),1)
        
        self.low_hz_ = self.add_weight(shape=(self.filters,),
                                       initializer=low_hz_init,
                                       name='low_hz')
        
        self.band_hz_ = self.add_weight(shape=(self.filters,),
                                       initializer=band_hz_init,
                                       name='band_hz')
        
        # Hamming window
        n_lin = tf.linspace(0., self.kernel_size, self.kernel_size)
        self.window_ = 0.54 - 0.46 * tf.cos(2*math.pi * n_lin / self.kernel_size)
 
        # n's to traverse
        n = (self.kernel_size - 1) / 2
        n_ = tf.range(-n, n+1, dtype='float') / self.sample_rate
        
        self.n_ = tf.expand_dims(n_, 0)

        self.built = True
        
    def sinc(self, x):
        # Similar to numpy implementation
        # Could optimise by only computing one half (symmetric)
        x = tf.where(tf.abs(x) < 1e-20, 1e-20 * tf.ones_like(x), x)
        return tf.sin(x) / x
    
    def call(self, inputs):
        '''
        Input shape: (batch_size, 1, n_samples)
        '''
        low = self.min_low_hz / self.sample_rate + tf.abs(self.low_hz_)
        high = low + self.min_band_hz / self.sample_rate + tf.abs(self.band_hz_)

        # Compute low_pass and high_pass sincs
        f_times_t = tf.matmul(low, self.n_)
        
        low_pass1 = 2 * low * self.sinc(
                    2 * math.pi * f_times_t * self.sample_rate)

        f_times_t = tf.matmul(high, self.n_)
        low_pass2 = 2 * high * self.sinc(
                    2 * math.pi * f_times_t * self.sample_rate)
      
        band_pass = low_pass2 - low_pass1

        # norm by max in each filter
        max_ = tf.reduce_max(band_pass, axis=1, keepdims=True)
        band_pass = band_pass / max_

        self.kernel = (band_pass * self.window_)
        self.kernel = tf.transpose(self.kernel, [1, 0])
        self.kernel = tf.expand_dims(self.kernel, 1)

        outputs = K.conv1d(inputs,
                           self.kernel,
                           strides=self.stride,
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate)
        
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
   
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
        elif self.data_format == 'channels_first':
            space = input_shape[2:]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                dilation=self.dilation_rate)
            new_space.append(new_dim)
        if self.data_format == 'channels_last':
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        elif self.data_format == 'channels_first':
            return (input_shape[0], self.filters) + tuple(new_space)
    
    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'initializer': self.initializer,
            'activation': activations.serialize(self.activation)
        }
        base_config = super(SincConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
