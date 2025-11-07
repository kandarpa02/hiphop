from tensorflow.python.framework.dtypes import DType, float32
import tensorflow as tf
from hiphop.typing import Initializer
from .module import Module
from .._src.layers.Linear import Linear
from .._src import functional
from typing import Sequence

def name_error(name:str):
    if hasattr(functional, name.lower()):
        return f"No such activation function '{name}', did you mean '{name.lower()}'?"
    else:
        f"No such activation function '{name}'"

def act(name):
    try:
        return getattr(functional, name)
    except AttributeError:
        raise AttributeError(name_error(name))
    

class _Linear(Module):
    def __init__(
            self, out_feat: int, bias: bool = True, weight_init: Initializer | None = None, 
            bias_init: Initializer | None = None, dtype: DType = tf.float32, name: str | None = None
            ):
        
        super().__init__(name)
        self.out_feat = out_feat
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.dtype = dtype
        self.linear = None

    def __call__(self, x):
        if self.linear is None:
            self.linear = Linear(
                x.shape[-1], self.out_feat,
                weight_init=self.weight_init,
                bias_init=self.bias_init,
                dtype=self.dtype
            )
        return self.linear(x)

class MLP(Module):
    def __init__(self, units:Sequence[int], nonlin:str='default', weight_init: Initializer | None = None, dtype: DType = tf.float32, name=None):
        super().__init__(name)
        self.nonlin = act('relu') if nonlin=='default' else act(nonlin)
        self.units = units
        self.weight_init = weight_init
        self.dtype = dtype
        self.layers = self._make_layers()

    def _make_layers(self):
        layers = []
        for i, u in enumerate(self.units):
            layer = _Linear(u, weight_init=self.weight_init, dtype=self.dtype)
            layers.append(layer)

        return layers

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            if i!=len(self.layers)-1:
                x = self.nonlin(layer(x))
            
            else:
                x = layer(x)

        return x