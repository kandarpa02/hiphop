VARIABLE_REGISTRY = {}
SCOPE_STACK = []
from ..typing import Initializer
from .._api import export
import tensorflow as tf
from typing import Any

def clear_scope():
    VARIABLE_REGISTRY = {}
    SCOPE_STACK = []

class variable_scope:
    def __init__(self, name, reuse=False, reset=False, dtype='float32'):
        self.name = name
        self.reuse = reuse
        self.reset = reset
        self.dtype = dtype

    def __enter__(self):
        effective_reuse = self.reuse or any(r for _, r in SCOPE_STACK)
        SCOPE_STACK.append((self.name, effective_reuse))

        if self.reset:
            prefix = "/".join(scope for scope, _ in SCOPE_STACK)
            keys_to_remove = [k for k in VARIABLE_REGISTRY if k.startswith(prefix)]
            for k in keys_to_remove:
                del VARIABLE_REGISTRY[k]

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        SCOPE_STACK.pop()

def variables_in_scope(scope_name):
    return {k: v for k, v in VARIABLE_REGISTRY.items() if k.startswith(scope_name + "/")}

def get_variable(name: str, shape=None, initializer:Initializer=lambda: None, dtype:tf.DType=tf.float32, rng=None):

    shape = shape or []

    full_scope = "/".join(scope for scope, _ in SCOPE_STACK) if SCOPE_STACK else ""
    full_name = f"{full_scope}/{name}" if full_scope else name

    # Determine if reuse is allowed from current scope stack
    reuse_allowed = any(r for _, r in SCOPE_STACK)

    # Variable already exists
    if full_name in VARIABLE_REGISTRY:
        if not reuse_allowed:
            raise ValueError(f"Variable {full_name} already exists, but reuse is False")
        return VARIABLE_REGISTRY[full_name]

    # Variable does not exist but reuse is requested
    if reuse_allowed:
        raise ValueError(f"Variable {full_name} does not exist, cannot reuse")

    # Otherwise, create a new variable
    try:
        data = initializer(shape=shape, dtype=dtype, key=rng)
    except TypeError:
        data = initializer(shape=shape, dtype=dtype)

    out = tf.Variable(data, trainable=True, name=name)
    VARIABLE_REGISTRY[full_name] = out
    return out 