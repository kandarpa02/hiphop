from typing import Dict, Any
import tensorflow as tf


class Module(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)

    def _variable_naming(self, prefix=""):
        return f"{prefix}{self.name}/" if self.name else prefix

    def get_variable(
        self,
        name: str,
        shape=None,
        initializer=lambda **_: None,
        dtype: tf.DType = tf.float32,
        rng=None,
        prefix=""
    ):
        full_name = f"{self._variable_naming(prefix)}{name}"
        shape = shape or []
        try:
            data = initializer(shape=shape, dtype=dtype, key=rng)
        except TypeError:
            data = initializer(shape=shape, dtype=dtype)
        var = tf.Variable(data, dtype=dtype, trainable=True)
        setattr(self, name, var)
        return var
    
    def __call__(self, *args):
        raise NotImplementedError
    