from ._api import build_api
build_api()

from ._src.base import variable_scope, variables_in_scope, get_variable, clear_scope, build
from ._src.module.base import Module
# from ._src.module.transform import inline
from ._src.initializers.initializer import VarianceScaling, TruncatedNormal, Constant

from ._src.layers.functional.dense import dense
from ._src.layers.Linear import Linear
from ._src.layers.TensorOps import Flatten
from ._src.backward import valgrad, grad, jit_compile