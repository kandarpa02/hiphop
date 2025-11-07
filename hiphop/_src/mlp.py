from tensorflow.python.framework.dtypes import DType, float32
import tensorflow as tf
from hiphop.typing import Initializer
from .module import Module
from .._src.layers.Linear import Linear
from .._src import functional
from typing import Sequence, Optional, Iterable


def name_error(name: str) -> str:
    """Constructs a helpful error message for invalid activation names.

    Args:
        name: The provided activation function name.

    Returns:
        A formatted string suggesting the closest valid alternative.
    """
    if hasattr(functional, name.lower()):
        return f"No such activation function '{name}', did you mean '{name.lower()}'?"
    else:
        return f"No such activation function '{name}'"


def act(name: str):
    """Retrieves an activation function from the `functional` module.

    Args:
        name: The name of the activation function (e.g., `'relu'`, `'tanh'`, `'sigmoid'`).

    Returns:
        A callable activation function.

    Raises:
        AttributeError: If the specified activation function is not found.
    """
    try:
        return getattr(functional, name)
    except AttributeError:
        raise AttributeError(name_error(name))


class _Linear(Module):
    """A deferred-initialization wrapper around `Linear` for dynamic input dimensions.

    This helper layer lazily instantiates a `Linear` transformation on the
    first forward pass, inferring the input dimension from the input tensor.

    Example:
        ```python
        x = tf.random.normal([8, 64])
        layer = _Linear(32)
        y = layer(x)  # Instantiates Linear(64, 32)
        ```

    Args:
        out_feat: Number of output features.
        bias: Whether to include a bias term. Defaults to True.
        weight_init: Optional initializer for weights.
        bias_init: Optional initializer for bias.
        dtype: TensorFlow data type. Defaults to `tf.float32`.
        name: Optional string name for the module.
    """

    def __init__(
        self,
        out_feat: int,
        bias: bool = True,
        weight_init: Initializer | None = None,
        bias_init: Initializer | None = None,
        dtype: DType = tf.float32,
        initial_shape:int=None,
        name: str | None = None,
    ):
        """Initializes the `_Linear` layer parameters."""
        super().__init__(name)
        self.out_feat = out_feat
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.dtype = dtype
        self.linear = None
        self.init_shape = initial_shape
        if self.init_shape is not None:
            self.linear = Linear(
                    self.init_shape,
                    self.out_feat,
                    weight_init=self.weight_init,
                    bias_init=self.bias_init,
                    dtype=self.dtype,
                )

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """Applies a linear transformation to the input.

        Instantiates an internal `Linear` layer on first use, using the
        input's last dimension as `in_feat`.

        Args:
            x: Input tensor of shape `[..., in_feat]`.

        Returns:
            Output tensor of shape `[..., out_feat]`.
        """
        if self.linear is None:
            self.linear = Linear(
                x.shape[-1],
                self.out_feat,
                weight_init=self.weight_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
            )

        return self.linear(x)

class MLP(Module):
    """A standard multilayer perceptron (feed-forward neural network).

    This module constructs a stack of `Linear` layers separated by a
    user-specified activation function. The final layer is left linear
    (no activation) by convention.

    Example:
        ```python
        import hiphop as hh
        import tensorflow as tf

        # Create a 3-layer MLP: 784 → 128 → 64 → 10
        mlp = hh.MLP([784, 128, 64, 10], nonlin='relu')

        x = tf.random.normal([32, 784])
        y = mlp(x)  # shape: (32, 10)
        ```

    Args:
        units: Sequence of layer sizes, including input and output dimensions.
            For example, `[784, 128, 64, 10]` defines a 3-layer MLP.
        nonlin: Name of activation function (e.g., `'relu'`, `'tanh'`, `'sigmoid'`).
            Use `'default'` to select ReLU.
        weight_init: Optional initializer for all layer weights.
        dtype: TensorFlow data type for parameters. Defaults to `tf.float32`.
        name: Optional string name for variable scoping.
    """

    def __init__(
        self,
        units: Sequence[int],
        nonlin: str = "default",
        weight_init: Optional[Initializer] = None,
        dtype: DType = tf.float32,
        name: Optional[str] = None,
    ):
        """Initializes all submodules of the MLP."""
        super().__init__(name)
        self.nonlin = act("relu") if nonlin == "default" else act(nonlin)
        self.units = units
        self.weight_init = weight_init
        self.dtype = dtype
        self.layers = self._make_layers()

    def _make_layers(self) -> list[Linear]:
        """Constructs the sequence of `Linear` layers.

        Returns:
            A list of `Linear` modules connecting consecutive dimensions in `units`.
        """
        layers: list[Linear] = []
        for in_feat, out_feat in zip(self.units[:-1], self.units[1:]):
            layer = Linear(in_feat, out_feat, weight_init=self.weight_init, dtype=self.dtype)
            layers.append(layer)
        return layers

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """Applies the multilayer perceptron to the input tensor.

        Each hidden layer applies a linear transformation followed by
        the selected activation function. The final layer is linear.

        Args:
            x: Input tensor of shape `[..., units[0]]`.

        Returns:
            Output tensor of shape `[..., units[-1]]`.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.nonlin(x)
        return x