import numpy as np
from base import Base_Layer
from dense import parse_dtype
from sequential import Sequential
from tensor import Tensor


class BaseModel(Base_Layer):
    def __init__(self):
        super().__init__()
        self._modules = {}  # Tracks all child components
        self.device = 'cpu'
        self.dtype = np.float32  # Default dtype

    def __setattr__(self, name, value):
        """Auto-register layers/modules when assigned as attributes"""
        if not name.startswith('_'):
            if isinstance(value, (Base_Layer, Sequential)) or \
               (hasattr(value, 'parameters') and hasattr(value, 'forward')):
                if not hasattr(self, '_modules'):
                    self._modules = {}
                self._modules[name] = value
        super().__setattr__(name, value)

    @property
    def parameters(self):
        """Collect all trainable parameters across all submodules"""
        params = []
        for module in self._modules.values():
            if hasattr(module, 'parameters'):
                params += module.parameters
            elif isinstance(module, Tensor):
                params.append(module)
        return params

    def forward(self, inputs):
        """Default forward pass through registered modules in assignment order"""
        x = inputs
        for name, module in self._modules.items():
            x = module(x)
        return x

    def set_device(self, device):
        """Propagate device setting to all subcomponents"""
        self.device = device
        for module in self._modules.values():
            if hasattr(module, 'set_device'):
                module.set_device(device)
            elif hasattr(module, 'device'):
                module.device = device
            if isinstance(module, Tensor):
                module.to(device)

    def set_dtype(self, dtype):
        """Cast all parameters to specified dtype"""
        self.dtype = dtype
        for param in self.parameters:
            if hasattr(param, 'astype'):
                param.data = param.data.astype(dtype)

    def state_dict(self):
        """Aggregate state dicts from all submodules"""
        state = {
            '_meta': {
                'device': self.device,
                'dtype': str(self.dtype)
            }
        }
        for name, module in self._modules.items():
            if hasattr(module, 'state_dict'):
                state[name] = module.state_dict()
            elif isinstance(module, Tensor):
                state[name] = module.data.copy()
        return state

    def load_state_dict(self, state_dict):
        """Load state dicts into appropriate submodules"""
        meta = state_dict.pop('_meta', {})
        self.set_device(meta.get('device', 'cpu'))
        self.set_dtype(parse_dtype(meta.get('dtype', np.float32)))

        for name, data in state_dict.items():
            module = getattr(self, name)
            if hasattr(module, 'load_state_dict'):
                module.load_state_dict(data)
            elif isinstance(module, Tensor):
                module.data = data

    def zero_grad(self):
        """Clear gradients from all parameters"""
        for param in self.parameters:
            param.zero_grad()
    def no_grad(self):
        """Convenience method to access Tensor's no_grad context"""
        return Tensor.no_grad()


    def __call__(self, inputs):
        return self.forward(inputs)

    def __repr__(self):
        return f"{self.__class__.__name__}(\n" + \
               "\n".join(f"  ({name}): {module}"
                        for name, module in self._modules.items()) + "\n)"