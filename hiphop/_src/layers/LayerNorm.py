import tensorflow as tf
from ..module import Module
from typing import Tuple
from ..initializers.initializer import Constant

class LayerNorm(Module):
    """
    Implements Layer Normalization.

    Normalizes input across the given `axis` dimensions for each sample,
    maintaining mean=0 and variance=1 within each feature group, then applies
    learnable scale (gamma) and shift (beta).

    Args:
        axis: int or tuple of ints, the feature dimension(s) to normalize over.
        epsilon: small float added to variance to avoid dividing by zero.
        dtype: data type for parameters.
        name: optional module name.
    """
    def __init__(self, axis=-1, epsilon=1e-5, dtype: tf.DType = tf.float32, name=None):
        super().__init__(name)
        if isinstance(axis, int):
            self.axis = (axis,)
        elif isinstance(axis, tuple):
            self.axis = axis
        else:
            raise ValueError("axis should be int or tuple of int")

        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
        self.dtype = dtype

    def __call__(self, x):
        # Lazily create scale/shift parameters
        if self.gamma is None or self.beta is None:
            # For layer norm, param_shape is based on the *normalized dimensions*
            param_shape = [x.shape[a] for a in self.axis]
            param_shape = tf.TensorShape(param_shape)

            self.gamma = self.get_variable(
                "gamma", shape=param_shape, initializer=Constant(1.0), dtype=self.dtype
            )
            self.beta = self.get_variable(
                "beta", shape=param_shape, initializer=Constant(0.0), dtype=self.dtype
            )

        mean, var = tf.nn.moments(x, axes=self.axis, keepdims=True)
        x_hat = (x - mean) / tf.sqrt(var + self.epsilon)

        # Broadcast gamma and beta correctly
        out = x_hat * self.gamma + self.beta
        return tf.cast(out, self.dtype)
