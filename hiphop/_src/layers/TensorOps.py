from ..._src.module.base import Module
from ..initializers.initializer import Constant, VarianceScaling
import tensorflow as tf

class Flatten(Module):
    def __init__(self, preserve_dims=1, name=None):
        super().__init__(name)
        self.dim = preserve_dims

    def __call__(self, x):
        shape = x.shape
        pre = shape[:self.dim]
        next = shape[len(pre):]
        Shape = pre + (int(tf.reduce_prod(next)),)
        return tf.reshape(x, Shape)