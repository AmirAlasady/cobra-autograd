import numpy as np
from collections.abc import Iterable

# Conditional CuPy import
try:
    import cupy as cp # type: ignore
    has_cupy = True
except ImportError:
    cp = None
    has_cupy = False

def unbroadcast_grad(grad, shape, xp):
    """Sum gradients over broadcasted dimensions."""
    if grad.shape == shape:
        return grad

    # Handle added dimensions
    ndims_added = grad.ndim - len(shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)

    # Handle dimensions with size 1
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad

class Tensor:
    EPSILON = 1e-12
    _no_grad_mode = False  # Class-level flag
    @classmethod
    def no_grad(cls):
        """Context manager to disable gradient tracking"""
        class NoGradContext:
            def __enter__(self):
                cls._no_grad_mode = True

            def __exit__(self, exc_type, exc_val, exc_tb):
                cls._no_grad_mode = False

        return NoGradContext()
    def __init__(self, data, device='cpu', dtype=np.float32, requires_grad=False):
        self.device = device
        self.dtype = dtype
        if Tensor._no_grad_mode:
            self.requires_grad = False
        else:
            self.requires_grad = requires_grad
        self._op = None
        self._prev = set()
        self._backward = lambda: None
        self._pre_backward_hooks = []
        self.xp = np if device == 'cpu' else cp

        # Initialize data
        if isinstance(data, self.xp.ndarray):
            self.data = data.astype(dtype)
        else:
            self.data = self.xp.array(data, dtype=dtype)

        # Device consistency check
        if device == 'cuda' and not has_cupy:
            raise RuntimeError("CuPy not installed. Cannot use device='cuda'.")

        # Initialize gradient
        self.grad = None
        # Initialize gradient with same dtype as tensor
        if self.requires_grad:
            self.grad = Tensor(
                self.xp.zeros_like(self.data, dtype=self.dtype),  # <-- Add explicit dtype
                device=self.device,
                dtype=self.dtype,
                requires_grad=False
            )
    def astype(self, dtype):
        return Tensor(
            self.xp.array(self.data, dtype=dtype),
            device=self.device,
            dtype=dtype,
            requires_grad=self.requires_grad
        )
    def __getitem__(self, indices):
        """Enable slicing/indexing of tensor data"""
        if isinstance(indices, Tensor):
            if indices.device != self.device:
                indices = indices.to(self.device)
            indices = indices.data
        out_data = self.data[indices]
        out = Tensor(out_data,
                    device=self.device,
                    dtype=self.dtype,
                    requires_grad=self.requires_grad)

        if self.requires_grad:
            out._prev = {self}
            def _backward():
                grad = self.xp.zeros_like(self.data)
                grad[indices] = out.grad.data
                self.grad.data += grad
            out._backward = _backward
        return out

    def gather(self, dim, index):
        """Gather values along specified dimension using index tensor"""
        assert dim == 1, "Currently only supports dim=1 for this implementation"

        # Ensure index is integer type
        if not isinstance(index, Tensor) or index.dtype not in (np.int32, np.int64):
            index = index.astype(np.int64) if isinstance(index, Tensor) else \
                    Tensor(index.data.astype(np.int64), device=self.device)

        # Create output tensor
        out_data = self.xp.take_along_axis(self.data, index.data, axis=dim)
        out = Tensor(out_data, device=self.device, dtype=self.dtype,
                    requires_grad=self.requires_grad)

        if self.requires_grad:
            out._prev = {self, index}
            def _backward():
                if self.requires_grad:
                    # Create zero-initialized gradient tensor
                    grad = self.xp.zeros_like(self.data)
                    
                    # Create indices for all dimensions
                    indices = list(self.xp.indices(index.data.shape))
                    
                    # Replace target dimension with index values
                    indices[dim] = index.data
                    
                    # Assign gradients using advanced indexing
                    grad[tuple(indices)] = out.grad.data
                    
                    # Accumulate gradients
                    if self.grad is None:
                        self.grad = Tensor(grad, dtype=self.dtype, device=self.device)
                    else:
                        self.grad.data += grad
            out._backward = _backward

        return out

    def argmax(self, axis=None, keepdims=False):
        """Returns indices of maximum values along an axis"""
        out_data = self.xp.argmax(self.data, axis=axis)
        if keepdims:
            out_data = self.xp.expand_dims(out_data, axis=axis)

        # Argmax is non-differentiable, so new tensor has requires_grad=False
        return Tensor(out_data,
                     device=self.device,
                     dtype=self.dtype,#np.int64
                     requires_grad=False)

    def reshape(self, *shape):
        """Return new tensor with reshaped data"""
        out_data = self.xp.reshape(self.data, shape)
        out = Tensor(out_data, device=self.device, dtype=self.dtype,
                    requires_grad=self.requires_grad)

        # Maintain computation graph
        if self.requires_grad:
            out._prev = {self}

            def _backward():
                if self.requires_grad:
                    self.grad.data += out.grad.data.reshape(self.shape)
            out._backward = _backward

        return out
    def one_hot(self, num_classes):
        """Convert class indices to one-hot encoding (for cross-entropy)"""
        indices = self.data.astype(int)
        if self.device == 'cpu':
            one_hot = np.eye(num_classes)[indices]
        else:
            one_hot = cp.eye(num_classes)[indices]
        return Tensor(one_hot, device=self.device,
                      dtype=self.dtype,# remove this if needed
                      )

    def to(self, device):
        """Move tensor to specified device."""
        if self.device == device:
            return self

        new_xp = np if device == 'cpu' else cp
        if device == 'cpu':
            new_data = cp.asnumpy(self.data) if self.device == 'cuda' else self.data.copy()
        else:
            if not has_cupy:
                raise RuntimeError("CuPy not installed.")
            new_data = cp.asarray(self.data)

        return Tensor(new_data, device=device, dtype=self.dtype, requires_grad=self.requires_grad)

    def __ge__(self, other):
        # Ensure other is a Tensor with the same device and dtype
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype, requires_grad=False)
        xp = self.xp
        # Use the underlying array comparison
        result = xp.greater_equal(self.data, other.data)
        # Return a boolean Tensor (non-differentiable)
        return Tensor(result.astype(np.bool_), device=self.device, requires_grad=False)

    def __le__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype, requires_grad=False)
        xp = self.xp
        result = xp.less_equal(self.data, other.data)
        return Tensor(result.astype(np.bool_), device=self.device, requires_grad=False)

    def clip(self, min_val, max_val):
        """
        Clip tensor values between min and max with proper gradient flow.
        The output will always have the same dtype as self.
        """
        xp = self.xp
        # Convert min_val and max_val to tensors with self.dtype
        min_val_cast = Tensor(xp.array(min_val.data, dtype=self.dtype),
                              device=self.device, requires_grad=False)
        max_val_cast = Tensor(xp.array(max_val.data, dtype=self.dtype),
                              device=self.device, requires_grad=False)

        # For each element:
        #   if self >= min_val_cast then keep self, else use min_val_cast.
        lower = self.where(self >= min_val_cast, min_val_cast)
        # Then, if lower <= max_val_cast then keep lower, else use max_val_cast.
        clipped = lower.where(lower <= max_val_cast, max_val_cast)

        return clipped


    def backward(self, grad=None):
        """Backpropagate gradients through computation graph with dtype checks"""
        # Handle gradient argument with dtype enforcement
        if grad is not None:
            # Convert to Tensor if needed
            if not isinstance(grad, Tensor):
                grad = Tensor(grad, device=self.device, dtype=self.dtype)
            else:
                # Cast to self's dtype if mismatch
                if grad.dtype != self.dtype:
                    grad = grad.astype(self.dtype)

                # Ensure same device
                if grad.device != self.device:
                    grad = grad.to(self.device)

            assert grad.dtype == self.dtype, \
                f"Gradient dtype {grad.dtype} must match tensor dtype {self.dtype}"
            assert grad.device == self.device, \
                f"Gradient device {grad.device} must match tensor device {self.device}"

            self.grad.data = grad.data.astype(self.grad.dtype)
        else:
            if self.data.size != 1:
                raise RuntimeError("backward() requires gradient argument for non-scalar tensors")
            self.grad.data = self.xp.ones_like(self.data)

        # Topological sort
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited and v.requires_grad:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Reverse-mode autograd with dtype checks
        for tensor in reversed(topo):
            # Perform gradient casting before _backward()
            if tensor.grad is not None and tensor.grad.dtype != tensor.dtype:
                tensor.grad = tensor.grad.astype(tensor.dtype)

            tensor._backward()

            # Now check after casting
            if tensor.grad is not None:
                assert tensor.grad.dtype == tensor.dtype, \
                    f"Gradient dtype {tensor.grad.dtype} != tensor dtype {tensor.dtype} (post-cast)"


    def zero_grad(self):
        """Reset gradients to zero."""
        if self.grad is not None:
            self.grad.data.fill(0)

    def register_hook(self, hook):
        """Register gradient hook."""
        self._pre_backward_hooks.append(hook)
        # Add comparison operators
    def __gt__(self, other):
        return self._compare(other, lambda a, b: a > b)

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b)

    def _compare(self, other, op):
        """Helper for comparison operators"""
        xp = self.xp
        if isinstance(other, Tensor):
            data = op(self.data, other.data)
        else:
            data = op(self.data, other)

        return Tensor(
            data.astype(np.bool_),  # Convert to boolean array
            device=self.device,
            requires_grad=False,
            dtype=self.dtype,# remove this if needed
            )

    def where(self, condition, y):
        """Element-wise conditional: condition ? self : y.

        This method ensures that the tensor `y` is cast to the dtype of `self`
        (i.e. self.dtype) before performing the element-wise operation.
        """
        xp = self.xp
        # Ensure condition is a boolean Tensor.
        if not isinstance(condition, Tensor):
            condition = Tensor(
                condition,
                device=self.device,
                dtype=np.bool_,  # Force boolean dtype.
                requires_grad=False
            )

        # Ensure y is a Tensor and that its dtype matches self.dtype.
        if not isinstance(y, Tensor):
            y = Tensor(
                y,
                device=self.device,
                dtype=self.dtype,  # Enforce self's dtype.
                requires_grad=False
            )
        else:
            # If y is already a Tensor but has a different dtype, cast it.
            if y.dtype != self.dtype:
                # Convert y.data to the calling tensor's dtype.
                y = Tensor(
                    xp.array(y.data, dtype=self.dtype),
                    device=self.device,
                    requires_grad=y.requires_grad,
                    dtype=self.dtype,  # Enforce self's dtype.
                )

        # Perform the element-wise 'where' operation.
        out_data = xp.where(condition.data, self.data, y.data)
        out = Tensor(
            out_data,
            device=self.device,
            requires_grad=(self.requires_grad or y.requires_grad),
            dtype=self.dtype  # Output is created with self.dtype.
        )
        out._prev = {self, y}
        out._op = 'where'
        # Save condition as a boolean array for the backward pass.
        out._saved_condition = condition.data.astype(xp.bool_)

        def _backward():
            # Gradient for self (applied where condition is True).
            if self.requires_grad:
                grad_self = out.grad.data * out._saved_condition.astype(self.dtype)
                self.grad.data += unbroadcast_grad(grad_self, self.shape, xp)
            # Gradient for y (applied where condition is False).
            if y.requires_grad:
                grad_y = out.grad.data * (~out._saved_condition).astype(self.dtype)
                y.grad.data += unbroadcast_grad(grad_y, y.shape, xp)

        out._backward = _backward
        return out

    # --------------------------
    # Core Operations (Fixed)
    # --------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device,
                                                               dtype=self.dtype,#remove this if needed
                                                               )
        assert self.device == other.device, "Devices must match"

        out_data = self.data + other.data
        out = Tensor(out_data, device=self.device, requires_grad=(self.requires_grad or other.requires_grad),
                     dtype=self.dtype,#remove this if needed
                     )
        out._prev = {self, other}
        out._op = 'add'

        def _backward():
            xp = out.xp
            if self.requires_grad:
                self.grad.data += unbroadcast_grad(out.grad.data, self.shape, xp)
            if other.requires_grad:
                other.grad.data += unbroadcast_grad(out.grad.data, other.shape, xp)
        out._backward = _backward
        return out

    def __mul__(self, other):
        
        if isinstance(other, Tensor): 
            other.device = self.device
            other.dtype = self.dtype
        else:
            other=Tensor(other, device=self.device,dtype=self.dtype)

        assert self.device == other.device, "Devices must match"

        out_data = self.data * other.data
        out = Tensor(out_data, device=self.device, requires_grad=(self.requires_grad or other.requires_grad),
                     dtype=self.dtype,#remove this if needed
                     )
        out._prev = {self, other}
        out._op = 'mul'

        def _backward():
            xp = out.xp
            if self.requires_grad:
                self.grad.data += unbroadcast_grad(other.data * out.grad.data, self.shape, xp)
            if other.requires_grad:
                other.grad.data += unbroadcast_grad(self.data * out.grad.data, other.shape, xp)
        out._backward = _backward
        return out
    
    """ old code
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device,
                    dtype=self.dtype,#remove this if needed
                    )
        assert self.device == other.device, "Devices must match"

        out_data = self.data @ other.data
        #  ---- or use out_data = xp.matmul(self.data, other.data) for clearaty ----

        out = Tensor(out_data, device=self.device, requires_grad=(self.requires_grad or other.requires_grad),
                    dtype=self.dtype,#remove this if needed
                    )
        out._prev = {self, other}
        out._op = 'matmul'

        def _backward():
            xp = out.xp
            if self.requires_grad:
                self.grad.data += xp.matmul(out.grad.data, other.data.T)
            if other.requires_grad:
                other.grad.data += xp.matmul(self.data.T, out.grad.data)
        out._backward = _backward
        return out
    """
    # new code with batch dimension
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype)
        assert self.device == other.device, "Devices must match"

        out_data = self.data @ other.data
        out = Tensor(out_data, device=self.device, requires_grad=(self.requires_grad or other.requires_grad),
                    dtype=self.dtype)
        out._prev = {self, other}
        out._op = 'matmul'

        def _backward():
            xp = out.xp
            # Handle batch dimensions by summing over batch axis
            if self.requires_grad:
                grad_self = xp.einsum('...ij,...kj->...ik', out.grad.data, other.data)
                self.grad.data += grad_self.sum(axis=tuple(range(grad_self.ndim - 2)))
                
            if other.requires_grad:
                grad_other = xp.einsum('...ki,...kj->...ij', self.data, out.grad.data)
                other.grad.data += grad_other.sum(axis=tuple(range(grad_other.ndim - 2)))

        out._backward = _backward
        return out
    
    def square(self):
        """Element-wise square operation"""
        return self ** 2  # Leverage existing __pow__ method

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "Exponent must be scalar"
        out_data = self.data ** exponent
        out = Tensor(out_data, device=self.device, requires_grad=self.requires_grad,
                     dtype=self.dtype,#remove this if needed
                     )
        out._prev = {self}
        out._op = f'pow_{exponent}'

        def _backward():
            if self.requires_grad:
                self.grad.data += (exponent * self.data ** (exponent - 1)) * out.grad.data
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
      """Sum elements along specified axis"""
      out_data = self.xp.sum(self.data, axis=axis, keepdims=keepdims)
      out = Tensor(out_data, device=self.device, requires_grad=self.requires_grad,
                   dtype=self.dtype,#remove this if needed
                  )
      if self.requires_grad:
          out._prev = {self}

          def _backward():
              grad = out.grad.data

              # Handle dimension expansion for proper broadcasting
              if axis is not None and not keepdims:
                  grad = self.xp.expand_dims(grad, axis=axis)

              # Broadcast gradient to original shape
              grad = self.xp.broadcast_to(grad, self.data.shape)
              self.grad.data += grad

          out._backward = _backward

      return out

    """ old code
    def mean(self, axis=None):
        num_elements = np.prod(self.data.shape) if axis is None else self.data.shape[axis]
        denom = self.xp.array(num_elements, dtype=self.dtype)
        return self.sum(axis=axis) / float(denom)
    """
    def mean(self, axis=None, keepdims=False):
        """Compute mean with axis support"""
        out_data = self.xp.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, 
                    device=self.device,
                    dtype=self.dtype,
                    requires_grad=self.requires_grad)

        if self.requires_grad:
            out._prev = {self}
            def _backward():
                if axis is None:
                    grad = self.xp.ones_like(self.data) * out.grad.data / self.data.size
                else:
                    if keepdims:
                        grad = out.grad.data / self.data.shape[axis]
                    else:
                        grad = self.xp.expand_dims(out.grad.data, axis=axis) / self.data.shape[axis]
                    grad = self.xp.broadcast_to(grad, self.data.shape)
                self.grad.data += grad
            out._backward = _backward
            
        return out
# --
    def var(self, axis=None, keepdims=False, unbiased=True):
        """Compute variance with axis support"""
        mean = self.mean(axis=axis, keepdims=True)
        squared_diff = (self - mean).square()
        ddof = 1 if unbiased else 0
        if axis is None:
            n = self.data.size
        else:
            n = self.data.shape[axis]
        out_data = squared_diff.data.mean(axis=axis, keepdims=keepdims) * n / (n - ddof)
        
        out = Tensor(out_data,
                    device=self.device,
                    dtype=self.dtype,
                    requires_grad=self.requires_grad)

        if self.requires_grad:
            out._prev = {self}
            def _backward():
                # Gradient of variance calculation
                grad = (2 / (n - ddof)) * (self.data - mean.data) * out.grad.data
                if axis is not None and not keepdims:
                    grad = self.xp.expand_dims(grad, axis=axis)
                grad = self.xp.broadcast_to(grad, self.data.shape)
                self.grad.data += grad
            out._backward = _backward
            
        return out
# --
    def sqrt(self):
        """Element-wise square root"""
        return self ** 0.5
    
    def __truediv__(self, other):
        xp = self.xp
        eps_val = 1e-8 if self.dtype == np.float32 else 1e-16
        eps = xp.array(eps_val, dtype=self.dtype)
        if isinstance(other, Tensor):
            return self * (other + eps).reciprocal()
        else:
            # Create a Tensor from the scalar, ensuring the correct dtype.
            other_tensor = Tensor(xp.array(other, dtype=self.dtype), device=self.device, requires_grad=False,dtype=self.dtype)
            return self * (1.0 / (other_tensor.data + eps))

    def reciprocal(self):
        xp = self.xp
        eps_val = 1e-8 if self.dtype == np.float32 else 1e-16
        eps = xp.array(eps_val, dtype=self.dtype)
        one = xp.array(1.0, dtype=self.dtype)
        out_data = one / (self.data + eps)
        out = Tensor(out_data, device=self.device, requires_grad=self.requires_grad, dtype=self.dtype)
        out._prev = {self}
        out._op = 'reciprocal'

        def _backward():
            if self.requires_grad:
                self.grad.data += (-out_data ** 2) * out.grad.data
        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        """Compute maximum along specified axis with type consistency."""
        xp = self.xp
        # Compute maximum along the specified axis.
        out_data = xp.max(self.data, axis=axis, keepdims=keepdims)
        # Determine if gradients need to flow.
        requires_grad = self.requires_grad and xp.any(self.data == out_data)
        # Create output tensor with self.dtype.
        out = Tensor(out_data, device=self.device, requires_grad=requires_grad, dtype=self.dtype)

        if self.requires_grad:
            out._prev = {self}
            def _backward():
                # Build a mask of maximum elements and cast to self.dtype.
                mask = (self.data == out_data).astype(self.dtype)
                if axis is not None:
                    mask =mask / xp.sum(mask, axis=axis, keepdims=keepdims)
                # Propagate gradients only through the elements equal to the max.
                self.grad.data += mask * out.grad.data
            out._backward = _backward
        return out

    def exp(self):
        """Element-wise exponential with type consistency."""
        xp = self.xp
        # Compute exp; the result will inherit the dtype of self.data.
        out_data = xp.exp(self.data)
        out = Tensor(out_data, device=self.device, requires_grad=self.requires_grad, dtype=self.dtype)

        if self.requires_grad:
            out._prev = {self}
            def _backward():
                # Derivative of exp is exp itself.
                self.grad.data += out.data * out.grad.data
            out._backward = _backward
        return out

    def log(self):
        """Element-wise natural logarithm with type consistency."""
        xp = self.xp
        # Ensure EPSILON is cast to self.dtype.
        eps = xp.array(self.EPSILON, dtype=self.dtype)
        # Compute logarithm.
        out_data = xp.log(self.data + eps)
        out = Tensor(out_data, device=self.device, requires_grad=self.requires_grad, dtype=self.dtype)

        if self.requires_grad:
            out._prev = {self}
            def _backward():
                # The derivative of log is 1/(self.data + eps).
                self.grad.data += (1 / (self.data + eps)) * out.grad.data
            out._backward = _backward
        return out



    # --------------------------
    # Activation Functions
    # --------------------------
    def relu(self):
        mask = self.data > 0
        out_data = self.xp.where(mask, self.data, 0)
        out = Tensor(out_data, device=self.device, requires_grad=self.requires_grad,
                    dtype=self.dtype,#remove this if needed
                    )
        out._prev = {self}
        out._op = 'relu'
        out._saved_mask = mask

        def _backward():
            if self.requires_grad:
                self.grad.data += out._saved_mask * out.grad.data
        out._backward = _backward
        return out

    def sigmoid(self):
        out_data = 1 / (1 + self.xp.exp(-self.data))
        out = Tensor(out_data, device=self.device, requires_grad=self.requires_grad,
                    dtype=self.dtype,#remove this if needed
                    )
        out._prev = {self}
        out._op = 'sigmoid'
        out._saved_data = out_data

        def _backward():
            if self.requires_grad:
                self.grad.data += (out._saved_data * (1 - out._saved_data)) * out.grad.data
        out._backward = _backward
        return out

    def tanh(self):
        out_data = self.xp.tanh(self.data)
        out = Tensor(out_data, device=self.device, requires_grad=self.requires_grad,
                    dtype=self.dtype,#remove this if needed
                    )
        out._prev = {self}
        out._op = 'tanh'

        def _backward():
            if self.requires_grad:
                self.grad.data += (1 - out_data**2) * out.grad.data
        out._backward = _backward
        return out
    def softmax(self, axis=-1):
        # Shift for numerical stability
        shifted = self.data - self.xp.max(self.data, axis=axis, keepdims=True)
        exp_data = self.xp.exp(shifted)
        sum_exp = self.xp.sum(exp_data, axis=axis, keepdims=True)
        out_data = exp_data / sum_exp
        out = Tensor(out_data, device=self.device, requires_grad=self.requires_grad,
                    dtype=self.dtype,#remove this if needed
                    )
        out._prev = {self}
        out._op = 'softmax'

        def _backward():
            if self.requires_grad:
                # s: softmax output
                s = out_data
                # Compute dot product of (s * grad) along the specified axis
                dot = self.xp.sum(out.grad.data * s, axis=axis, keepdims=True)
                # The gradient of softmax: s * (grad - dot)
                self.grad.data += s * (out.grad.data - dot)
        out._backward = _backward
        return out

    def leaky_relu(self, negative_slope=0.01):
        out_data = self.xp.where(self.data > 0, self.data, self.data * negative_slope)
        out = Tensor(out_data, device=self.device, requires_grad=self.requires_grad,
                    dtype=self.dtype,#remove this if needed
                    )
        out._prev = {self}
        out._op = 'leaky_relu'
        def _backward():
            if self.requires_grad:
                grad_val = self.xp.where(self.data > 0, 1, negative_slope)
                self.grad.data += grad_val * out.grad.data
        out._backward = _backward
        return out
    # --------------------------
    # Other Methods
    # --------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        """Number of dimensions of the tensor"""
        return self.data.ndim

    def __repr__(self):
        return f"Tensor({self.data}, device='{self.device}', requires_grad={self.requires_grad})"

    @property
    def T(self):
        out_data = self.data.T
        out = Tensor(out_data, device=self.device, requires_grad=self.requires_grad,
                    dtype=self.dtype,#remove this if needed
                    )
        out._prev = {self}
        out._op = 'transpose'

        def _backward():
            if self.requires_grad:
                self.grad.data += out.grad.data.T
        out._backward = _backward
        return out


    ### conv ###
    def transpose(self, *axes):
        """Transpose dimensions with explicit axis ordering"""
        out_data = self.xp.transpose(self.data, axes)
        out = Tensor(out_data, device=self.device, dtype=self.dtype,
                    requires_grad=self.requires_grad)
        if self.requires_grad:
            out._prev = {self}
            def _backward():
                self.grad.data += self.xp.transpose(out.grad.data, axes)
            out._backward = _backward
        return out

    def pad2d(self, padding):
        """2D padding with gradient support"""
        xp = self.xp
        pad_width = ((0,0), (0,0), (padding, padding), (padding, padding))
        out_data = xp.pad(self.data, pad_width, mode='constant')
        out = Tensor(out_data, device=self.device, dtype=self.dtype,
                    requires_grad=self.requires_grad)

        if self.requires_grad:
            out._prev = {self}
            def _backward():
                if padding == 0:
                    self.grad.data += out.grad.data
                else:
                    self.grad.data += out.grad.data[:, :, padding:-padding, padding:-padding]
            out._backward = _backward

        return out


    @staticmethod
    def concatenate(tensors, axis=0):
        """
        Concatenates a list of Tensors along the specified axis.
        Gradients are split and passed back to the original tensors.
        """
        # Use the numerical library (np or cp) from the first tensor.
        xp = tensors[0].xp
        data_list = [t.data for t in tensors]
        out_data = xp.concatenate(data_list, axis=axis)
        requires_grad = any(t.requires_grad for t in tensors)
        device = tensors[0].device
        dtype = tensors[0].dtype
        out = Tensor(out_data, device=device, dtype=dtype, requires_grad=requires_grad)

        def _backward():
            grad = out.grad.data
            # Determine the sizes along the concatenation axis.
            sizes = [t.data.shape[axis] for t in tensors]
            indices = np.cumsum(sizes)[:-1]
            grad_splits = xp.split(grad, indices, axis=axis)
            for t, g in zip(tensors, grad_splits):
                if t.requires_grad:
                    t.grad.data += g
        out._backward = _backward
        out._prev = set(tensors)
        return out

    @staticmethod
    def unsqueeze(tensor, axis):
        """
        Inserts a singleton dimension at the given axis.
        Its backward simply squeezes the gradient along that axis.
        """
        xp = tensor.xp
        new_shape = list(tensor.data.shape)
        new_shape.insert(axis, 1)
        out_data = xp.reshape(tensor.data, new_shape)
        out = Tensor(out_data, device=tensor.device, dtype=tensor.dtype, requires_grad=tensor.requires_grad)
        if tensor.requires_grad:
            def _backward():
                tensor.grad.data += xp.squeeze(out.grad.data, axis=axis)
            out._backward = _backward
            out._prev = {tensor}
        return out

    @staticmethod
    def stack(tensors, axis=0):
        """
        Stacks a list of Tensors along a new axis.
        Implemented by unsqueezing each tensor at the given axis and then concatenating.
        """
        tensors_unsqueezed = [Tensor.unsqueeze(t, axis) for t in tensors]
        return Tensor.concatenate(tensors_unsqueezed, axis=axis)

    @classmethod
    def randn(cls, *shape, device='cpu', dtype=np.float32):
        xp = np if device == 'cpu' else cp
        return cls(xp.random.randn(*shape).astype(dtype), 
                device=device, dtype=dtype, requires_grad=True)

    def copy(self):
        return Tensor(self.data.copy(), 
                    device=self.device,
                    dtype=self.dtype,
                    requires_grad=self.requires_grad)
    
    # Operator overloads
    __radd__ = __add__
    __rmul__ = __mul__
    __neg__ = lambda self: self * -1
    __sub__ = lambda self, other: self + (-other)
    __rtruediv__ = lambda self, other: Tensor(other) / self

from abc import ABC, abstractmethod

class Base_Layer(ABC):
    _id_counter = 0

    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.device = 'cpu'  # Default device
        self.id = f"{self.__class__.__name__}_{self.get_next_id()}"

    def get_next_id(self):
        Base_Layer._id_counter += 1
        return Base_Layer._id_counter

    def __call__(self, x):
        """Enable layer calling syntax: layer(input)"""
        return self.forward(x)

    def set_gpu(self):
        pass

    def set_cpu(self):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

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
            if x.device != self.device:  # Add this check
                x = x.to(self.device)
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


# Ensure you import your Tensor class appropriately
# from your_tensor_module import Tensor
from abc import ABC, abstractmethod

from base import Base_Layer
from tensor import Tensor
# Ensure you import your Tensor class appropriately
# from your_tensor_module import Tensor

class Activation(Base_Layer):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def set_gpu(self):
        # Consider using 'cuda' if that's what your Tensor expects.
        self.device = 'cuda'

    def set_cpu(self):
        self.device = 'cpu'

    @abstractmethod
    def forward(self, inputs):
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs, device=self.device, dtype=self.dtype)  # Add dtype
        elif inputs.device != self.device:
            inputs = inputs.to(self.device)
        return self._forward_impl(inputs)

class Tanh(Activation):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"

    def forward(self, inputs):
        # Ensure inputs are Tensor instances
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs, device=self.device)
        return inputs.tanh()

    def state_dict(self):
        return {"activation": type(self).__name__}

    def load_state_dict(self, state_dict):
        pass

    def __call__(self, x):
        return self.forward(x)

class ReLU(Activation):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"

    def forward(self, inputs):
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs, device=self.device)
        return inputs.relu()

    def state_dict(self):
        return {"activation": type(self).__name__}

    def load_state_dict(self, state_dict):
        pass

    def __call__(self, x):
        return self.forward(x)

class Sigmoid(Activation):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"

    def forward(self, inputs):
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs, device=self.device)
        return inputs.sigmoid()

    def state_dict(self):
        return {"activation": type(self).__name__}

    def load_state_dict(self, state_dict):
        pass

    def __call__(self, x):
        return self.forward(x)

class Softmax(Activation):
    def __init__(self,axis=-1, device='cpu'):
        super().__init__(device)
        self.axis = axis
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"
        # Optionally, add an axis parameter if needed:
        # self.axis = axis

    def forward(self, inputs):
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs, device=self.device)
        # Currently using axis=0; adjust if you want to apply softmax over a different axis.
        return inputs.softmax(axis=self.axis)

    def state_dict(self):
        return {"activation": type(self).__name__}

    def load_state_dict(self, state_dict):
        pass

    def __call__(self, x):
        return self.forward(x)

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01, device='cpu'):
        super().__init__(device)
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"
        assert alpha > 0, "alpha must be positive"
        self.alpha = alpha

    def forward(self, inputs):
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs, device=self.device)
        xp = inputs.xp  # Get numpy/cupy from the Tensor
        condition = inputs > 0
        # Use xp.expm1 for numerical stability
        return inputs.where(condition, self.alpha * xp.expm1(inputs.data))

    def state_dict(self):
        return {"activation": type(self).__name__, "alpha": self.alpha}

    def load_state_dict(self, state_dict):
        self.alpha = state_dict.get("alpha", self.alpha)

    def __call__(self, x):
        return self.forward(x)

class ELU(Activation):
    def __init__(self, alpha=1.0, device='cpu'):
        super().__init__(device)
        assert alpha > 0, "alpha must be positive"
        self.alpha = alpha

    def forward(self, inputs):
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs, device=self.device)
        xp = inputs.xp
        condition = inputs.data > 0
        return inputs.where(condition, self.alpha * (inputs.exp() - 1))

    def state_dict(self):
        return {"activation": type(self).__name__, "alpha": self.alpha}

    def load_state_dict(self, state_dict):
        self.alpha = state_dict.get("alpha", self.alpha)

    def __call__(self, x):
        return self.forward(x)

from tensor import Tensor


class Sequential:
    def __init__(self, layers, device='cpu'):
        """
        Initialize with a list of layers containing Tensor operations.
        """
        self.layers = layers
        self.device = device  # Default device

    @property
    def parameters(self):
        """Get all trainable parameters"""
        params = []
        for layer in self.layers:
            layer_params = getattr(layer, "parameters", None)
            if layer_params:
                params.extend(layer_params)
        return params

    def set_device(self, device):
        """Move all parameters to the specified device"""
        self.device = device
        for layer in self.layers:
            if hasattr(layer, 'set_device'):
                layer.set_device(device)
            elif hasattr(layer, 'parameters'):
                for param in layer.parameters:
                    if hasattr(param, "to"):  # Ensure the parameter supports `.to(device)`
                        param.to(device)

    def forward(self, x):
        """
        Forward propagation through all layers using Tensor operations.
        """
        if not isinstance(x, Tensor):
            x = Tensor(x, device=self.device)

        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def no_grad(self):
        """Disable gradient tracking for all parameters"""
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                for param in layer.parameters:
                    param.requires_grad = False

    def zero_grad(self):
        """Reset gradients for all parameters"""
        for param in self.parameters:
            param.zero_grad()

    def state_dict(self):
        """Return model state as dictionary of Tensors"""
        state = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'state_dict'):
                state[f'layer_{i}'] = layer.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Load model state from dictionary"""
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'load_state_dict'):
                layer.load_state_dict(state_dict.get(f'layer_{i}', {}))



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


import numpy as np
try:
    import cupy as cp # type: ignore
except ImportError:
    cp = None

class Optimizer:
    """Base optimizer class with learning rate decay"""
    def __init__(self, params, lr=0.01, decay=0.0):
        """
        Args:
            params: List of trainable parameters (Tensors)
            lr: Initial learning rate
            decay: Learning rate decay factor (per epoch)
        """
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.initial_lr = lr
        self.decay = decay
        self.iterations = 0
        self.xp = np  # Default to numpy

    def _get_xp(self, param):
        """Get correct numerical library for parameter"""
        return np if param.device == 'cpu' else cp # type: ignore

    def step(self):
        """Update parameters (to be implemented by subclasses)"""
        raise NotImplementedError

    def decay_lr(self):
        """Exponential learning rate decay"""
        self.lr *= (1.0 - self.decay)

class SGD(Optimizer):
    """Momentum SGD with learning rate decay"""
    def __init__(self, params, lr=0.01, momentum=0.9, decay=0.0):
        super().__init__(params, lr, decay)
        self.momentum = momentum
        self.velocities = [self._get_xp(p).zeros_like(p.data) for p in self.params]

    def step(self):
        self.iterations += 1
        for i, param in enumerate(self.params):
            xp = self._get_xp(param)

            # Update velocity
            self.velocities[i] = self.momentum * self.velocities[i] + \
                                (1 - self.momentum) * param.grad.data

            # Update parameters
            param.data -= self.lr * self.velocities[i]

class Adam(Optimizer):
    """Adam optimizer with learning rate decay"""
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, decay=0.0):
        super().__init__(params, lr, decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [self._get_xp(p).zeros_like(p.data) for p in self.params]
        self.v = [self._get_xp(p).zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            xp = self._get_xp(param)

            # Update first and second moment estimates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad.data
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * xp.square(param.grad.data)

            # Bias-corrected estimates
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            param.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.epsilon)



import numpy as np
from tensor import Tensor
try:
    import cupy as cp # type: ignore
except ImportError:
    cp = None


class Loss:
    """Base loss class with numerical stability safeguards"""
    EPSILON = 1e-12

    def __call__(self, pred: 'Tensor', target: 'Tensor') -> 'Tensor':
        raise NotImplementedError


class MSELoss(Loss):
    """Mean Squared Error Loss with automatic broadcasting"""
    def __call__(self, pred: 'Tensor', target: 'Tensor') -> 'Tensor':
        """
        Args:
            pred: Prediction tensor of shape (batch_size, ...)
            target: Target tensor of shape (batch_size, ...)

        Returns:
            MSE loss tensor with automatic broadcasting
        """
        return (pred - target).square().mean()


class CrossEntropy(Loss):
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        # Numerically stable implementation
        max_val = pred.max(axis=-1, keepdims=True)
        stable_exp = (pred - max_val).exp()
        softmax = stable_exp / stable_exp.sum(axis=-1, keepdims=True)  # Now works!
        return -(target * softmax.log()).mean()



class SoftmaxCrossEntropyLoss(Loss):
    def __call__(self, pred: 'Tensor', target: 'Tensor') -> 'Tensor':
        # If target has the same number of dimensions as pred,
        # assume it's one-hot encoded; otherwise, assume it's indices.
        if target.ndim == pred.ndim:
            class_indices = target.argmax(axis=-1).astype(np.int64)
        else:
            class_indices = target.astype(np.int64)

        # Numerical stability: compute log-softmax
        max_pred = pred.max(axis=-1, keepdims=True)
        log_sum_exp = (pred - max_pred).exp().sum(axis=-1, keepdims=True).log()
        log_softmax = pred - max_pred - log_sum_exp

        # Flatten log_softmax from shape (batch, seq_len, vocab_size) to (batch * seq_len, vocab_size)
        log_softmax_flat = log_softmax.reshape(-1, log_softmax.shape[-1])
        
        # Flatten class indices from shape (batch, seq_len) to (batch * seq_len, 1)
        class_indices = class_indices.reshape(-1, 1)

        # Gather the log probabilities for the correct classes along axis 1
        nll_loss = -log_softmax_flat.gather(1, class_indices)
        return nll_loss.mean()


class BCELoss(Loss):
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """Binary Cross Entropy with proper Tensor operations"""
        ones = Tensor(1.0)
        epsilon = Tensor(1e-7)

        # Clip using Tensor instances
        clipped = pred.clip(epsilon, ones - epsilon)
        return -(target * clipped.log() + (ones - target) * (ones - clipped).log()).mean()


class Accuracy:
    """Flexible accuracy metric supporting multiple formats"""
    def __call__(self, pred: 'Tensor', target: 'Tensor') -> float:
        """
        Args:
            pred: (batch_size, num_classes) logits/probabilities
            target: (batch_size,) class indices or (batch_size, num_classes) one-hot

        Returns:
            Accuracy score between 0 and 1
        """
        # Get prediction classes
        pred_classes = pred.argmax(axis=-1).data

        # Handle different target formats
        if target.ndim > 1 and target.shape[-1] > 1:
            # One-hot encoded targets
            target_classes = target.argmax(axis=-1).data
        else:
            # Class indices
            target_classes = target.data.squeeze()

        # Device-aware comparison
        if isinstance(pred_classes, np.ndarray):
            return np.mean(pred_classes == target_classes)
        else:  # cupy
            return float(cp.mean(pred_classes == target_classes))
        
import numpy as np
try:
    import cupy as cp # type: ignore
except ImportError:
    cp = None

class Optimizer:
    """Base optimizer class with learning rate decay"""
    def __init__(self, params, lr=0.01, decay=0.0):
        """
        Args:
            params: List of trainable parameters (Tensors)
            lr: Initial learning rate
            decay: Learning rate decay factor (per epoch)
        """
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.initial_lr = lr
        self.decay = decay
        self.iterations = 0
        self.xp = np  # Default to numpy

    def _get_xp(self, param):
        """Get correct numerical library for parameter"""
        return np if param.device == 'cpu' else cp # type: ignore

    def step(self):
        """Update parameters (to be implemented by subclasses)"""
        raise NotImplementedError

    def decay_lr(self):
        """Exponential learning rate decay"""
        self.lr *= (1.0 - self.decay)

class SGD(Optimizer):
    """Momentum SGD with learning rate decay"""
    def __init__(self, params, lr=0.01, momentum=0.9, decay=0.0):
        super().__init__(params, lr, decay)
        self.momentum = momentum
        self.velocities = [self._get_xp(p).zeros_like(p.data) for p in self.params]

    def step(self):
        self.iterations += 1
        for i, param in enumerate(self.params):
            xp = self._get_xp(param)

            # Update velocity
            self.velocities[i] = self.momentum * self.velocities[i] + \
                                (1 - self.momentum) * param.grad.data

            # Update parameters
            param.data -= self.lr * self.velocities[i]

class Adam(Optimizer):
    """Adam optimizer with learning rate decay"""
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, decay=0.0):
        super().__init__(params, lr, decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [self._get_xp(p).zeros_like(p.data) for p in self.params]
        self.v = [self._get_xp(p).zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            xp = self._get_xp(param)

            # Update first and second moment estimates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad.data
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * xp.square(param.grad.data)

            # Bias-corrected estimates
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            param.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.epsilon)


class BaseModel:
    def __init__(self):
        """Initialize BaseModel with empty modules dictionary."""
        self._modules = {}  # Tracks all child components
        self.device = 'cpu'
        self.dtype = np.float32  # Default dtype

    def __setattr__(self, name, value):
        """
        Auto-register layers/modules AND parameters when assigned as attributes.
        Fixed to handle framework's specific classes.
        """
        # First register appropriate elements in _modules
        if not name.startswith('_'):
            # Register specific types that framework uses
            if (isinstance(value, (Base_Layer, BaseModel, Sequential, Conv2D)) or 
                (isinstance(value, Tensor) and hasattr(value, 'requires_grad'))):
                
                # Ensure _modules exists
                if not hasattr(self, '_modules'):
                    self._modules = {}
                    
                # Register in _modules dictionary
                self._modules[name] = value
        
        # Always set the attribute using parent method
        super().__setattr__(name, value)

    @property
    def parameters(self):
        """
        Collect all trainable parameters across all submodules.
        Handles framework's parameter collection logic.
        """
        params = []
        for module in self._modules.values():
            # For modules with their own parameters property/method
            if hasattr(module, 'parameters'):
                if callable(getattr(module, 'parameters')) and not isinstance(module.parameters, property):
                    # It's a method - call it
                    params.extend(module.parameters())
                else:
                    # It's a property - access it directly
                    params.extend(module.parameters)
            # For direct Tensor parameters
            elif isinstance(module, Tensor) and module.requires_grad:
                params.append(module)
        return params

    def forward(self, inputs):
        """
        Default forward pass through registered modules in assignment order.
        Simple sequential processing.
        """
        x = inputs
        for name, module in self._modules.items():
            if hasattr(module, '__call__'):
                x = module(x)
        return x

    def set_device(self, device):
        """
        Propagate device setting to all subcomponents.
        Fixed to handle device migration in your framework.
        """
        self.device = device
        
        # Update each registered module
        for name in list(self._modules.keys()):
            module = self._modules[name]
            
            # Handle direct Tensor attributes
            if isinstance(module, Tensor):
                moved = module.to(device)
                self._modules[name] = moved
                setattr(self, name, moved)
            # Handle modules with set_device method
            elif hasattr(module, 'set_device'):
                module.set_device(device)
            # Handle other modules with device attribute
            elif hasattr(module, 'device'):
                module.device = device
                # Update parameters in layers like Dense
                if hasattr(module, 'parameters'):
                    # Try common parameter names
                    for param_name in ['weights', 'bias']:
                        if hasattr(module, param_name):
                            param = getattr(module, param_name)
                            if hasattr(param, 'to'):
                                moved_param = param.to(device)
                                setattr(module, param_name, moved_param)

    def set_dtype(self, dtype):
        """
        Cast all parameters to specified dtype.
        Fixed to work with tensor data rather than the tensor objects.
        """
        # Store dtype in model
        self.dtype = dtype
        
        # Cast tensor data in parameters
        for param in self.parameters:
            if hasattr(param, 'data') and hasattr(param.data, 'astype'):
                param.data = param.data.astype(dtype)

    def state_dict(self):
        """
        Aggregate state dicts from all submodules.
        Fixed to properly handle tensor data.
        """
        state = {
            '_meta': {
                'device': self.device,
                'dtype': str(self.dtype)
            }
        }
        
        # Process each module
        for name, module in self._modules.items():
            if hasattr(module, 'state_dict') and callable(module.state_dict):
                # Handle modules with their own state_dict
                state[name] = module.state_dict()
            elif isinstance(module, Tensor) and hasattr(module, 'data'):
                # Handle direct tensor data
                state[name] = module.data.copy()
        
        return state

    def load_state_dict(self, state_dict):
        """
        Load state dicts into appropriate submodules.
        Fixed for framework's parameter handling.
        """
        # Extract meta info if available
        if '_meta' in state_dict:
            meta = state_dict.pop('_meta')
            target_device = meta.get('device', 'cpu')
            
            # Handle dtype parsing safely
            if 'dtype' in meta:
                try:
                    target_dtype = parse_dtype(meta.get('dtype'))
                    self.dtype = target_dtype
                except Exception:
                    # Fall back to default
                    pass
                
            # Set device globally first
            self.device = target_device
        
        # Load each module's state
        for name, data in state_dict.items():
            if hasattr(self, name):
                module = getattr(self, name)
                
                if hasattr(module, 'load_state_dict') and callable(module.load_state_dict):
                    # Handle modules with load_state_dict method
                    module.load_state_dict(data)
                elif isinstance(module, Tensor) and hasattr(module, 'data'):
                    # Handle direct tensors
                    module.data = data
                    
                    # Ensure correct device
                    if hasattr(module, 'device') and module.device != self.device:
                        setattr(self, name, module.to(self.device))
        
        # Apply dtype update
        self.set_dtype(self.dtype)

    def zero_grad(self):
        """Clear gradients from all parameters."""
        for param in self.parameters:
            if hasattr(param, 'zero_grad'):
                param.zero_grad()

    def no_grad(self):
        """Convenience method to access Tensor's no_grad context."""
        return Tensor.no_grad()

    def __call__(self, inputs):
        """Enable direct calling of model instances."""
        return self.forward(inputs)

    def __repr__(self):
        """Pretty string representation of the model structure."""
        return f"{self.__class__.__name__}(\n" + \
               "\n".join(f"  ({name}): {module}"
                        for name, module in self._modules.items()) + "\n)"




class Embedding(BaseModel):
    def __init__(self, vocab_size, d_model, dtype=np.float32):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dtype = dtype
        # Xavier initialization using parent's device/dtype
        init_std = np.sqrt(2.0 / (vocab_size + d_model))
        weight_data = np.random.randn(vocab_size, d_model).astype(self.dtype) * init_std
        
        self.weight = Tensor(weight_data, 
                           device=self.device,  # Use BaseModel's device
                           dtype=self.dtype,
                           requires_grad=True)

    def forward(self, x):
        # Ensure input is Tensor and on correct device
        if not isinstance(x, Tensor):
            x = Tensor(x, device=self.device, dtype=np.int64)
        else:
            x = x.to(self.device).astype(np.int64)  # Force device/dtype
        return self.weight[x]
    


class PositionalEncoding(BaseModel):
    def __init__(self, d_model: int, max_seq_len: int = 5000, dtype=np.float32):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pos_enc = self._create_positional_encoding()
        self.dtype = dtype

    def _create_positional_encoding(self):
        position = np.arange(self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(-(np.arange(0, self.d_model, 2) * np.log(10000.0) / self.d_model))
        
        pos_enc = np.zeros((self.max_seq_len, self.d_model), dtype=self.dtype)
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)
        
        return Tensor(pos_enc, 
                     device=self.device,
                     requires_grad=False,
                     dtype=self.dtype)

    def forward(self, x: Tensor) -> Tensor:
        # Get proper numerical library
        xp = x.xp
        
        # Get positional encoding for sequence length
        seq_len = x.shape[1]
        pos_enc = self.pos_enc[:seq_len]
        
        # Reshape positional encoding for broadcasting
        # From (seq_len, d_model) to (1, seq_len, d_model)
        pos_enc = pos_enc.reshape(1, seq_len, self.d_model)
        
        # Expand to match batch dimension
        # Result shape: (batch_size, seq_len, d_model)
        pos_enc = xp.broadcast_to(pos_enc.data, (x.shape[0], seq_len, self.d_model))
        
        return x + Tensor(pos_enc, device=self.device, dtype=self.dtype)

class MultiHeadAttention(BaseModel):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, dtype=np.float32):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.dtype = dtype
        # All layers inherit device/dtype from parent
        self.Wq = Dense(d_model, d_model,dtype=self.dtype)
        self.Wk = Dense(d_model, d_model,dtype=self.dtype)
        self.Wv = Dense(d_model, d_model,dtype=self.dtype)
        self.Wo = Dense(d_model, d_model,dtype=self.dtype)

    def __call__(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        return self.forward(query, key, value, mask)

    def split_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)\
               .transpose(0, 2, 1, 3)

    def combine_heads(self, x: Tensor) -> Tensor:
        return x.transpose(0, 2, 1, 3)\
               .reshape(x.shape[0], -1, self.d_model)

    def scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None):
        attn_scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.where(mask, -1e9)
        attn_weights = Softmax(axis=-1)(attn_scores)
        return attn_weights @ v

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
        q = self.split_heads(self.Wq(query))
        k = self.split_heads(self.Wk(key))
        v = self.split_heads(self.Wv(value))
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        return self.Wo(self.combine_heads(attn_output))



class LayerNorm(BaseModel):
    def __init__(self, features: int, eps: float = 1e-5, dtype=np.float32):
        """
        Layer Normalization
        
        Args:
            features: Size of the input feature dimension
            eps: Small value to prevent division by zero
            dtype: Data type for parameters
        """
        super().__init__()
        self.features = features
        self.eps = eps
        self.dtype = dtype

        # Learnable parameters
        self.gamma = Tensor(np.ones(features), 
                           dtype=dtype,
                           requires_grad=True)
        self.beta = Tensor(np.zeros(features),
                          dtype=dtype,
                          requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # Ensure input is on correct device
        x = x.to(self.device).astype(self.dtype)
        
        # Calculate statistics
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True, unbiased=False)  # Use biased variance
        
        # Normalize
        x_normalized = (x - mean) / (var + self.eps).sqrt()
        #print('====================>')
        #print(self.gamma.dtype)
        #print(self.gamma.device)
        #print(x_normalized.dtype)
        #print(x_normalized.device)
        # Scale and shift
        return self.gamma * x_normalized + self.beta

    @property
    def parameters(self):
        """Return gamma and beta for optimization"""
        return [self.gamma, self.beta]

    def state_dict(self):
        return {
            "gamma": self.gamma.data.copy(),
            "beta": self.beta.data.copy(),
            "features": self.features,
            "eps": self.eps,
            "device": self.device
        }

    def load_state_dict(self, state_dict):
        self.gamma.data = state_dict['gamma']
        self.beta.data = state_dict['beta']
        self.features = state_dict.get('features', self.features)
        self.eps = state_dict.get('eps', self.eps)


class Encoder(BaseModel):
    def __init__(self, vocab_size, d_model, num_heads, dtype=np.float32):
        super().__init__()
        self.dtype = dtype
        self.embeddings = Embedding(vocab_size, d_model, dtype=self.dtype)
        self.positional = PositionalEncoding(d_model, dtype=self.dtype)
        self.attention = MultiHeadAttention(d_model, num_heads, dtype=self.dtype)
        self.norm1 = LayerNorm(d_model, dtype=self.dtype)
        self.norm2 = LayerNorm(d_model, dtype=self.dtype)
        self.feedforward = Sequential([
            Dense(d_model, d_model * 4, dtype=self.dtype),
            ReLU(),
            Dense(d_model * 4, d_model, dtype=self.dtype)])

    def forward(self, x: Tensor) -> Tensor:
        # Embeddings and positional encoding
        x = self.embeddings(x)
        x = self.positional(x)
        
        # Attention block
        residual = x
        x = self.attention(x, x, x)
        x = residual + x
        x = self.norm1(x)
        
        # Feedforward block
        residual = x
        x = self.feedforward(x)
        x = residual + x
        x = self.norm2(x)
        
        return x
    


class Decoder(BaseModel):
    def __init__(self, vocab_size, d_model, num_heads, dtype=np.float32):
        super().__init__()
        self.dtype = dtype
        self.embeddings = Embedding(vocab_size, d_model, dtype=self.dtype)
        self.positional = PositionalEncoding(d_model, dtype=self.dtype)
        
        # Self attention with masking
        self.self_attention = MultiHeadAttention(d_model, num_heads, dtype=self.dtype)
        self.norm1 = LayerNorm(d_model, dtype=self.dtype)
        
        # Encoder-Decoder attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dtype=self.dtype)
        self.norm2 = LayerNorm(d_model, dtype=self.dtype)
        
        # Feedforward network
        self.feedforward = Sequential([
            Dense(d_model, d_model * 4, dtype=self.dtype),
            ReLU(),
            Dense(d_model * 4, d_model, dtype=self.dtype)
        ])
        self.norm3 = LayerNorm(d_model, dtype=self.dtype)

    def create_causal_mask(self, seq_len):
        """Create mask to prevent looking at future positions"""
        xp = np if self.device == 'cpu' else cp
        mask = xp.ones((1, seq_len, seq_len), dtype=self.dtype)
        mask = xp.tril(mask)  # Lower triangular matrix
        return Tensor(mask, device=self.device, requires_grad=False)

    def __call__(self, target: Tensor, encoder_output: Tensor) -> Tensor:
        return self.forward(target, encoder_output)

    def forward(self, target: Tensor, encoder_output: Tensor) -> Tensor:
        # Embed target sequence
        x = self.embeddings(target)
        x = self.positional(x)
        
        # Self attention with causal masking
        residual = x
        seq_len = x.shape[1]
        mask = self.create_causal_mask(seq_len)
        x = self.self_attention(x, x, x, mask)
        x = residual + x
        x = self.norm1(x)
        
        # Encoder-decoder attention
        residual = x
        x = self.cross_attention(x, encoder_output, encoder_output)
        x = residual + x
        x = self.norm2(x)
        
        # Feedforward block
        residual = x
        x = self.feedforward(x)
        x = residual + x
        x = self.norm3(x)
        
        return x




class Transformer(BaseModel):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, dtype=np.float32):
        """
        A full Transformer model that combines an Encoder, a Decoder, and a final projection layer.
        
        Args:
            src_vocab_size (int): Vocabulary size for the encoder (source language).
            tgt_vocab_size (int): Vocabulary size for the decoder (target language).
            d_model (int): Dimensionality of the model.
            num_heads (int): Number of attention heads.
            dtype: Data type for computations.
        """
        super().__init__()
        self.dtype = dtype

        # Build the encoder and decoder blocks (each includes embeddings and positional encoding)
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, dtype=dtype)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, dtype=dtype)
        
        # Final linear layer to map decoder output to target vocabulary logits
        self.final_linear = Dense(d_model, tgt_vocab_size, dtype=dtype)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        Forward pass of the Transformer.
        
        Args:
            src (Tensor): Input tensor of shape (batch_size, src_seq_len) containing token indices.
            tgt (Tensor): Target tensor of shape (batch_size, tgt_seq_len) containing token indices.
            
        Returns:
            Tensor: Logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Pass source tokens through the encoder to get a continuous representation.
        encoder_output = self.encoder(src)
        
        # Pass target tokens and the encoder output through the decoder.
        decoder_output = self.decoder(tgt, encoder_output)
        
        # Project the decoder output into the target vocabulary space.
        logits = self.final_linear(decoder_output)
        return logits

    def __call__(self, src: Tensor, tgt: Tensor) -> Tensor:
        return self.forward(src, tgt)


# functions for saving and loading state dictionaries as .pkl

import pickle


# saving function 
def save_model_parameters(object_to_save, file_path):
    print('<===== Saving =====>')
    state_dict = object_to_save.state_dict()
    with open(file_path, 'wb') as f:
        pickle.dump(state_dict, f)
        print('<===== Done =====>')
        
# loading function
def load_state_dict_from_file(file_path):
    with open(file_path, 'rb') as f:
        state_dict = pickle.load(f)
    return state_dict


