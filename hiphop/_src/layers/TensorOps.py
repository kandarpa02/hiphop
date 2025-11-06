from ..._src.module.base import Module
from ..initializers.initializer import Constant, VarianceScaling
import tensorflow as tf

class Flatten(Module):
    def __init__(self, preserve_dims=1, name=None):
        super().__init__(name)
        self.dim = preserve_dims

    def __call__(self, x):
        shape = tf.shape(x)
        pre = shape[:self.dim]
        next = shape[self.dim:]
        new_shape = tf.concat([pre, [tf.reduce_prod(next)]], axis=0)
        return tf.reshape(x, new_shape)