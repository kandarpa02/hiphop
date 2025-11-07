import tensorflow as tf
from ..module import Module
from ..initializers.initializer import VarianceScaling, Constant
from typing import Optional, Union
from ...typing import Initializer


def he_uniform():
    """He-uniform initializer (Kaiming uniform) for ReLU-family activations."""
    return VarianceScaling(scale=2.0, mode="fan_in", distribution="uniform")


# --------------------------------------------------------
# Conv1D
# --------------------------------------------------------
class Conv1d(Module):
    """Applies a 1D convolution over an input signal."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        pad: Union[str, int] = "same",
        bias: bool = True,
        weight_init: Optional[Initializer] = None,
        bias_init: Optional[Initializer] = None,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.stride = stride
        self.pad = pad
        self.bias = bias

        self.w_init = weight_init or he_uniform()
        self.w = self.get_variable(
            "w", [kernel_size, in_channels, out_channels], self.w_init, dtype=dtype
        )

        if bias:
            self.b_init = bias_init or Constant(0.0)
            self.b = self.get_variable("b", [out_channels], self.b_init, dtype=dtype)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        if isinstance(self.pad, str):
            y = tf.nn.conv1d(x, self.w, stride=self.stride, padding=self.pad.upper())
        elif isinstance(self.pad, int):
            y = tf.pad(x, [[0, 0], [self.pad, self.pad], [0, 0]])
            y = tf.nn.conv1d(y, self.w, stride=self.stride, padding="VALID")
        else:
            raise ValueError(f"Invalid pad type: {type(self.pad)}")

        if self.bias:
            y = y + self.b
        return y


# --------------------------------------------------------
# Conv2D
# --------------------------------------------------------
class Conv2d(Module):
    """Applies a 2D convolution over an input image."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        pad: Union[str, tuple[int, int], int] = "same",
        bias: bool = True,
        weight_init: Optional[Initializer] = None,
        bias_init: Optional[Initializer] = None,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.stride = stride
        self.pad = pad
        self.bias = bias

        self.w_init = weight_init or he_uniform()
        self.w = self.get_variable(
            "w",
            [kernel_size[0], kernel_size[1], in_channels, out_channels],
            self.w_init,
            dtype=dtype,
        )

        if bias:
            self.b_init = bias_init or Constant(0.0)
            self.b = self.get_variable("b", [out_channels], self.b_init, dtype=dtype)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        if isinstance(self.pad, str):
            y = tf.nn.conv2d(
                x,
                self.w,
                strides=[1, self.stride[0], self.stride[1], 1],
                padding=self.pad.upper(),
            )
        else:
            # handle tuple or int pad
            if isinstance(self.pad, int):
                pad_h = pad_w = self.pad
            elif isinstance(self.pad, (tuple, list)) and len(self.pad) == 2:
                pad_h, pad_w = self.pad
            else:
                raise ValueError(f"Invalid pad format: {self.pad}")
            x = tf.pad(x, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            y = tf.nn.conv2d(
                x,
                self.w,
                strides=[1, self.stride[0], self.stride[1], 1],
                padding="VALID",
            )

        if self.bias:
            y = y + self.b
        return y


# --------------------------------------------------------
# Conv3D
# --------------------------------------------------------
class Conv3d(Module):
    """Applies a 3D convolution over a volumetric input."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int] = (1, 1, 1),
        pad: Union[str, tuple[int, int, int], int] = "same",
        bias: bool = True,
        weight_init: Optional[Initializer] = None,
        bias_init: Optional[Initializer] = None,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.stride = stride
        self.pad = pad
        self.bias = bias

        self.w_init = weight_init or he_uniform()
        self.w = self.get_variable(
            "w",
            [kernel_size[0], kernel_size[1], kernel_size[2], in_channels, out_channels],
            self.w_init,
            dtype=dtype,
        )

        if bias:
            self.b_init = bias_init or Constant(0.0)
            self.b = self.get_variable("b", [out_channels], self.b_init, dtype=dtype)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        if isinstance(self.pad, str):
            y = tf.nn.conv3d(
                x,
                self.w,
                strides=[1, self.stride[0], self.stride[1], self.stride[2], 1],
                padding=self.pad.upper(),
            )
        else:
            # handle int or tuple
            if isinstance(self.pad, int):
                pad_d = pad_h = pad_w = self.pad
            elif isinstance(self.pad, (tuple, list)) and len(self.pad) == 3:
                pad_d, pad_h, pad_w = self.pad
            else:
                raise ValueError(f"Invalid pad format: {self.pad}")

            x = tf.pad(x, [[0, 0], [pad_d, pad_d], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            y = tf.nn.conv3d(
                x,
                self.w,
                strides=[1, self.stride[0], self.stride[1], self.stride[2], 1],
                padding="VALID",
            )

        if self.bias:
            y = y + self.b
        return y
