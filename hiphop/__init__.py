from ._api import build_api
build_api()

from ._src.base import variable_scope, variables_in_scope, get_variable, clear_scope
from ._src.initializers.initializer import VarianceScaling, TruncatedNormal, Constant