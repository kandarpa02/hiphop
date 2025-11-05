from typing import Dict, Any
import tensorflow as tf

class Module(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def get_variable(
        self,
        name: str,
        shape=None,
        initializer=lambda: None,
        dtype: tf.DType = tf.float32,
        rng=None
    ):
        shape = shape or []
        try:
            data = initializer(shape=shape, dtype=dtype, key=rng)
        except TypeError:
            data = initializer(shape=shape, dtype=dtype)
        return tf.Variable(data, trainable=True, name=name)


    def state_dict(self, prefix="") -> Dict[str, Any]:
        params = {}
        for name, attr in self.__dict__.items():
            if isinstance(attr, tf.Variable):
                params[f"{prefix}{name}"] = attr.numpy()
            elif isinstance(attr, Module):
                params.update(attr.state_dict(prefix=f"{prefix}{name}/"))
        return params

    def load_state_dict(self, params: Dict[str, Any]):
        for key, value in params.items():
            parts = key.split('/')
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            var_name = parts[-1]
            var = getattr(obj, var_name)
            var.assign(value)

    # def load_state_dict(self, params: Dict[str, Any]|Sequence):
    #     if isinstance(params, dict):
    #         for key, value in params.items():
    #             parts = key.split('/')
    #             obj = self
    #             for part in parts[:-1]:
    #                 obj = getattr(obj, part)
    #             var_name = parts[-1]
    #             var = getattr(obj, var_name)
    #             var.assign(value)
            
    #     else:
    #         self.variables = params

