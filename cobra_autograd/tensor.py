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