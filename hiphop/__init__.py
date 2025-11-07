import tensorflow as tf

_REQUIRED_MIN_VERSION = (2, 15)
_REQUIRED_MAX_VERSION = (2, 21)

def _check_tf_version():
    version = tuple(map(int, tf.__version__.split('.')[:2]))
    if not (_REQUIRED_MIN_VERSION <= version < _REQUIRED_MAX_VERSION):
        raise ImportError(
            f"HipHop requires TensorFlow >= {_REQUIRED_MIN_VERSION[0]}.{_REQUIRED_MIN_VERSION[1]} "
            f"and < {_REQUIRED_MAX_VERSION[0]}.{_REQUIRED_MAX_VERSION[1]}, "
            f"but found {tf.__version__}"
        )

_check_tf_version()

from ._src.module import Module
from ._src.Sequential import Sequential
from ._src.mlp import MLP

from ._src.initializers.initializer import VarianceScaling, TruncatedNormal, Constant
from ._src.functional import *

from ._src.layers.Activations import *
from ._src.layers.Linear import Linear
from ._src.layers.Conv import Conv1d, Conv2d, Conv3d
from ._src.layers.BatchNorm import BatchNorm1d, BatchNorm2d, BatchNorm3d
from ._src.layers.LayerNorm import LayerNorm
from ._src.layers.TensorOps import Flatten

from ._src.backward import valgrad, grad, jit_compile