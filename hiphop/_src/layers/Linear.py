from ..._src.module.base import Module
from ..initializers.initializer import Constant, VarianceScaling
import tensorflow as tf

class Linear(Module):
    def __init__(self, in_feat, out_feat, dtype=tf.float32, name=None):
        super().__init__(name)
        def he_uniform():
            return VarianceScaling(scale=2.0, mode="fan_in", distribution="uniform")
        self.w = self.get_variable('w' ,[in_feat, out_feat], he_uniform(), dtype=dtype)
        self.b = self.get_variable('b', [out_feat,], Constant(0.0), dtype=dtype)
    
    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b