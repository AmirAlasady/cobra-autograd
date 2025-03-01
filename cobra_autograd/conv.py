
import numpy as np
from base import Base_Layer
from tensor import Tensor


class Conv2D(Base_Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, device='cpu', dtype=np.float32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dtype = dtype

        # Xavier initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.kernels = Tensor(
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(dtype) * scale,
            device=device, dtype=dtype, requires_grad=True
        )
        self.bias = Tensor(
            np.zeros(out_channels, dtype=dtype),
            device=device, dtype=dtype, requires_grad=True
        )

    def forward(self, x):
        batch_size, in_channels, h_in, w_in = x.shape
        h_out = (h_in + 2*self.padding - self.kernel_size) // self.stride + 1
        w_out = (w_in + 2*self.padding - self.kernel_size) // self.stride + 1
        xp = x.xp

        # Pad input if needed
        if self.padding > 0:
            x = x.pad2d(self.padding)

        # Extract windows using stride tricks
        strides = (
            x.data.strides[0],  # Batch
            x.data.strides[1],  # Channels
            self.stride * x.data.strides[2],  # Height
            self.stride * x.data.strides[3],  # Width
            x.data.strides[2],  # Kernel height
            x.data.strides[3]   # Kernel width
        )

        windows = xp.lib.stride_tricks.as_strided(
            x.data,
            shape=(batch_size, self.in_channels, h_out, w_out, self.kernel_size, self.kernel_size),
            strides=strides
        )

        # Reshape for batch matrix multiplication
        x_col = windows.transpose(1, 4, 5, 0, 2, 3).reshape(
            self.in_channels * self.kernel_size * self.kernel_size,
            batch_size * h_out * w_out
        )

        # Reshape kernels for matrix multiplication
        k_col = self.kernels.data.reshape(self.out_channels, -1)

        # Matrix multiplication (most compute-intensive part)
        out = (k_col @ x_col).reshape(
            self.out_channels, batch_size, h_out, w_out
        ).transpose(1, 0, 2, 3)

        # Add bias
        out += self.bias.data.reshape(1, self.out_channels, 1, 1)

        return Tensor(out, device=x.device, dtype=self.dtype, requires_grad=x.requires_grad)

    def state_dict(self):
        return {"kernels": self.kernels.data.copy(), "bias": self.bias.data.copy(),
                "config": {"in_channels": self.in_channels, "out_channels": self.out_channels,
                           "kernel_size": self.kernel_size, "stride": self.stride, "padding": self.padding,
                           "dtype": str(self.dtype)}}
    def load_state_dict(self, state_dict):
        self.kernels.data = state_dict["kernels"]
        self.bias.data = state_dict["bias"]
    def __call__(self, x):
        """Explicit call interface (redundant but safe)"""
        return self.forward(x)

class MaxPool2D(Base_Layer):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x):
        xp = x.xp
        batch_size, channels, height, width = x.shape
        h_out = (height - self.kernel_size) // self.stride + 1
        w_out = (width - self.kernel_size) // self.stride + 1
        windows = xp.lib.stride_tricks.as_strided(
            x.data,
            shape=(batch_size, channels, h_out, w_out, self.kernel_size, self.kernel_size),
            strides=(x.data.strides[0], x.data.strides[1],
                     self.stride*x.data.strides[2], self.stride*x.data.strides[3],
                     x.data.strides[2], x.data.strides[3])
        )
        out_data = xp.max(windows, axis=(4,5))
        out = Tensor(out_data, device=x.device, dtype=x.dtype, requires_grad=x.requires_grad)
        # Save mask for backward (if needed)
        self.mask = (windows == out_data[..., None, None])
        return out
    def state_dict(self):
        return {"kernel_size": self.kernel_size, "stride": self.stride}
    def load_state_dict(self, state_dict):
        self.kernel_size = state_dict["kernel_size"]
        self.stride = state_dict["stride"]

class Flatten(Base_Layer):
    def __init__(self):
        super().__init__()
        self.original_shape = None
    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)
    def state_dict(self):
        return {}
    def load_state_dict(self, state_dict):
        pass