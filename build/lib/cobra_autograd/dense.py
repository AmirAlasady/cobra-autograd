import re
import numpy as np

from base import Base_Layer
from tensor import Tensor

def parse_dtype(dtype_str):
    """
    Convert a string representation of a NumPy dtype (or type) to the actual NumPy dtype.

    Works with inputs like:
      - "<class 'numpy.float32'>"
      - "float32"
      - "<class 'numpy.longdouble'>"
      - "longdouble"
      - "dtype('float64')"
    """
    s = str(dtype_str).strip()

    # Handle cases like "dtype('float64')"
    if s.startswith("dtype(") and s.endswith(")"):
        s = s[6:-1].strip("'\"")

    # Remove wrapping <class '...'> if present.
    s = s.replace("<class '", "").replace("'>", "")

    # Remove the "numpy." prefix if present.
    if s.startswith("numpy."):
        s = s[len("numpy."):]

    try:
        # np.dtype returns a dtype object. The `.type` attribute returns the corresponding scalar type.
        return np.dtype(s).type
    except Exception as e:
        raise ValueError(f"Invalid dtype string: {dtype_str}") from e

class Dense(Base_Layer):
    def __init__(self, input_size, output_size, name=None, initialization='xavier',
                 device='cpu', dtype=np.float32):  # ADDED dtype parameter
        super().__init__()
        self.name = name
        self.device = device
        self.initialization = initialization
        self.dtype = dtype  # Store dtype as instance variable
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"

        # Initialize parameters with specified dtype
        if self.initialization == 'xavier':
            init_std = np.sqrt(2.0 / (input_size + output_size))
            weight_data = np.random.randn(input_size, output_size).astype(dtype) * init_std  # CHANGED to dtype
        else:
            weight_data = np.random.randn(input_size, output_size).astype(dtype) * 0.01  # CHANGED to dtype

        self.weights = Tensor(weight_data, device=device, dtype=dtype, requires_grad=True)  # ADDED dtype
        self.bias = Tensor(np.zeros((1, output_size), dtype=dtype),  # CHANGED to dtype
                          device=device, dtype=dtype, requires_grad=True)  # ADDED dtype


    def set_device(self, device):
        """Move all layer parameters to specified device"""
        if self.device != device:
            self.weights = self.weights.to(device)
            self.bias = self.bias.to(device)
            self.device = device


    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, device=self.device, dtype=self.dtype)
        else:
            # Force cast to layer's dtype
            if x.dtype != self.dtype:
                x = x.astype(self.dtype)
        return x @ self.weights + self.bias

    @property
    def parameters(self):
        """Return trainable parameters as Tensors"""
        return [self.weights, self.bias]

    def state_dict(self):
        return {
            "weights": self.weights.data.copy(),
            "bias": self.bias.data.copy(),
            "device": self.device,
            "dtype": str(self.dtype)
        }

    def load_state_dict(self, state_dict):
        self.weights.data = state_dict['weights']
        self.bias.data = state_dict['bias']
        dtype_str = state_dict['dtype']
        self.dtype = parse_dtype(dtype_str)
        self.set_device(state_dict['device'])

    def __call__(self, x):
        return self.forward(x)

    def zero_grad(self):
        if self.weights.grad is not None:
            self.weights.grad.data.fill(0)  # Reset gradient in Tensor
        if self.bias.grad is not None:
            self.bias.grad.data.fill(0)  # Reset gradient in Tensor