import tensorflow as tf
from ..module import Module
from ..initializers.initializer import Constant
from typing import Optional


class _BatchNorm(Module):
    """Base class for 1D/2D/3D Batch Normalization."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Learnable parameters
        if affine:
            self.gamma = self.get_variable("gamma", [num_features], Constant(1.0), dtype)
            self.beta = self.get_variable("beta", [num_features], Constant(0.0), dtype)

        # Running statistics
        if track_running_stats:
            self.running_mean = tf.Variable(
                tf.zeros([num_features], dtype=dtype), trainable=False, name="running_mean"
            )
            self.running_var = tf.Variable(
                tf.ones([num_features], dtype=dtype), trainable=False, name="running_var"
            )
        else:
            self.running_mean = None
            self.running_var = None

    def _batch_norm(self, x, training: bool):
        # Determine which mean/variance to use
        if training:
            mean, var = tf.nn.moments(x, axes=self._reduce_axes, keepdims=False)
            if self.track_running_stats:
                # Update running statistics
                self.running_mean.assign(
                    (1 - self.momentum) * self.running_mean + self.momentum * mean
                )
                self.running_var.assign(
                    (1 - self.momentum) * self.running_var + self.momentum * var
                )
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_hat = (x - mean) / tf.sqrt(var + self.eps)

        # Apply affine transform if enabled
        if self.affine:
            x_hat = x_hat * self.gamma + self.beta

        return x_hat


# -------------------------------------------------------
# BatchNorm1d
# -------------------------------------------------------
class BatchNorm1d(_BatchNorm):
    """Applies Batch Normalization over 2D or 3D input (N, C) or (N, T, C)."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, dtype, name)
        self._reduce_axes = [0]  # batch axis only

    def __call__(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        # For [N, T, C], reduce over N and T
        if len(x.shape) == 3:
            self._reduce_axes = [0, 1]
        elif len(x.shape) == 2:
            self._reduce_axes = [0]
        else:
            raise ValueError(f"Unexpected input shape for BatchNorm1d: {x.shape}")

        return self._batch_norm(x, training)


# -------------------------------------------------------
# BatchNorm2d
# -------------------------------------------------------
class BatchNorm2d(_BatchNorm):
    """Applies Batch Normalization over 4D input (N, H, W, C)."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, dtype, name)
        self._reduce_axes = [0, 1, 2]

    def __call__(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        return self._batch_norm(x, training)


# -------------------------------------------------------
# BatchNorm3d
# -------------------------------------------------------
class BatchNorm3d(_BatchNorm):
    """Applies Batch Normalization over 5D input (N, D, H, W, C)."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, dtype, name)
        self._reduce_axes = [0, 1, 2, 3]

    def __call__(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        return self._batch_norm(x, training)
