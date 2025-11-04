from ._api import build_api
build_api()

from ._src.base import variable_scope, variables_in_scope, get_variable, clear_scope
from ._src.initializers.initializer import VarianceScaling, TruncatedNormal, Constant

from ._src.layers.functional.dense import dense
from ._src.backward import valgrad, grad, jit_compile