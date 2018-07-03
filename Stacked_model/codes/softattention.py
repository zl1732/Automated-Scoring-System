import numpy as np
import copy
import inspect
import types as python_types
import marshal
import sys
import warnings

from keras import backend as K
#import tensorflow
from keras import activations, initializers, regularizers
from keras.engine.topology import Layer, InputSpec
import numpy as np
#import theano
# from keras.layers.wrappers import Wrapper, TimeDistributed
# from keras.layers.core import Dense
# from keras.layers.recurrent import Recurrent, time_distributed_dense


# Build attention pooling layer
import numpy as np
import copy
import inspect
import types as python_types
import marshal
import sys
import warnings

from keras import backend as K
import tensorflow as tf
from keras import activations, initializers, regularizers
from keras.engine.topology import Layer, InputSpec
import numpy as np
#import theano
# from keras.layers.wrappers import Wrapper, TimeDistributed
# from keras.layers.core import Dense
# from keras.layers.recurrent import Recurrent, time_distributed_dense


# Build attention pooling layer
class Attention(Layer):
    # https://keras.io/layers/writing-your-own-keras-layers/
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        print("input shape: ",input_shape) # (None, 50, 100)
        init_val_v = (np.random.randn(input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_v = K.variable(init_val_v, name='att_v')
        init_val_W = (np.random.randn(input_shape[2], input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_W = K.variable(init_val_W, name='att_W')
        self.trainable_weights = [self.att_v, self.att_W]
    
    def call(self, x, mask=None):
        #print('x shape:', x.shape)
        #print('att_W:', self.att_W.shape)
        #print('att_v:',self.att_v.shape)
        y = K.dot(x, self.att_W)
        #print('y:',y.shape)
        #print("*"*20)
        if not self.activation:
            if K.backend() == 'theano':
                weights = K.theano.tensor.tensordot(self.att_v, y, axes=[0, 2])
            elif K.backend() == 'tensorflow':
                weights = K.tf.tensordot(self.att_v, y, axes=[0, 2])
        elif self.activation == 'tanh':
            if K.backend() == 'theano':
                weights = K.theano.tensor.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
            elif K.backend() == 'tensorflow':
                weights = K.tf.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
                #print('weight, after tensordot:',weights.shape)
        #print('weight, before softmax:',weights.shape)
        weights = K.softmax(weights)
        #print('weight, after softmax:',weights.shape)
        #print("*"*20)
        out = x * K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])
        #print('K.repeat(weights, x.shape[2]):',K.repeat(weights, x.shape[2]).shape)
        #print('K.permute_dimensions([0, 2, 1])',K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1]).shape)
        #print('out=x*permute:', out.shape)
        #print("*"*20)
        if self.op == 'attsum':
            #print('out, before sum:', out.shape)
            out = K.sum(out, axis=1)
            #print('out, after sum:', out.shape)
        elif self.op == 'attmean':
            out = K.sum(out,axis=1) / mask.sum(axis=1, keepdims=True)
        return K.cast(out, K.floatx())

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def compute_mask(self, x, mask):
        return None
    
    def get_config(self):
        config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

