from .module import Module
from typing import Optional, Callable, Iterable, Any

class Sequential(Module):
    """A container module that applies a sequence of layers in order.

    `Sequential` is a simple utility for chaining together multiple layers or
    modules, where the output of one layer is passed directly as the input
    to the next. It is especially convenient for defining feed-forward
    neural networks such as MLPs.

    Example:
        ```python
        import hiphop as hh
        import tensorflow as tf

        model = hh.Sequential([
            hh.Linear(784, 128),
            tf.nn.relu,
            hh.Linear(128, 64),
            tf.nn.relu,
            hh.Linear(64, 10),
        ])

        x = tf.random.normal([32, 784])
        y = model(x)  # shape: (32, 10)
        ```

    This container does **not** define its own parameters; instead, it holds
    references to submodules and their trainable variables.

    Notes:
    - If the first layer requires extra arguments (e.g., `(x, training=True)`),
      they can be passed through `Sequential.__call__` and will only be applied
      to the first module in the sequence.
    - Each subsequent layer receives only the output of the previous one.

    Args:
        layers: Optional iterable of callables or `Module` instances
            that will be applied in sequence.
        name: Optional name for the module scope.
    """

    def __init__(
        self,
        layers: Optional[Iterable[Callable[..., Any]]] = None,
        name: Optional[str] = None,
    ):
        """Initializes the sequential container and stores the given layers."""
        super().__init__(name=name)
        self._layers = list(layers) if layers is not None else []

    def __call__(self, inputs, *args, **kwargs):
        """Applies each layer in order to the input tensor.

        Args:
            inputs: Input tensor or structure passed to the first layer.
            *args: Additional positional arguments forwarded to the **first** layer only.
            **kwargs: Additional keyword arguments forwarded to the **first** layer only.

        Returns:
            The output of the final layer in the sequence.
        """
        outputs = inputs
        for i, mod in enumerate(self._layers):
            if i == 0:
                # Pass extra arguments only to the first module
                outputs = mod(outputs, *args, **kwargs)
            else:
                outputs = mod(outputs)
        return outputs