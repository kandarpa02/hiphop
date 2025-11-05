from ...._src.base import variable_scope, get_variable
from ...initializers.initializer import VarianceScaling, Constant
import tensorflow as tf

def he_uniform():
    return VarianceScaling(scale=2.0, mode="fan_in", distribution="uniform")

def dense(x, units, name, reset=True):
    with variable_scope(name, reset=reset) as scope:
        w = get_variable('w', [x.shape[-1], units], initializer=he_uniform())
        b = get_variable('b', [units,], initializer=Constant(0.0))
        return tf.matmul(x, w) + b
    
