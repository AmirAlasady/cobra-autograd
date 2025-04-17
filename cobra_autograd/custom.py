import numpy as np
from base import Base_Layer
from dense import parse_dtype
from sequential import Sequential
from tensor import Tensor
from conv import Conv2D

"""
class BaseModel:
    def __init__(self):
        super().__init__()
        self._modules = {}  # Tracks all child components
        self.device = 'cpu'
        self.dtype = np.float32  # Default dtype

   
    #def __setattr__(self, name, value):
    #   #Auto-register layers/modules when assigned as attributes
    #    if not name.startswith('_'):
    #        if isinstance(value, (Base_Layer, Sequential, Conv2D)) or \
    #           (hasattr(value, 'parameters') and hasattr(value, 'forward')):
    #            if not hasattr(self, '_modules'):
    #                self._modules = {}
    #            self._modules[name] = value
    #    super().__setattr__(name, value)
  
    def __setattr__(self, name, value):

        if not name.startswith('_'):
            # Register modules and tensors with requires_grad=True
            if isinstance(value, (Base_Layer, BaseModel, Sequential,Conv2D)) or \
               (isinstance(value, Tensor) and value.requires_grad) or \
                (isinstance(value, Tensor) and not value.requires_grad): 
                if not hasattr(self, '_modules'):
                    self._modules = {}
                self._modules[name] = value
                
        super().__setattr__(name, value)
    @property
    def parameters(self):
        
        params = []
        for module in self._modules.values():
            if hasattr(module, 'parameters'):
                params += module.parameters
            elif isinstance(module, Tensor):
                params.append(module)
        return params

    def forward(self, inputs):
        
        x = inputs
        for name, module in self._modules.items():
            x = module(x)
        return x

    def set_device(self, device):
        
        self.device = device
        # Iterate over a copy of keys to allow modification during iteration
        for name in list(self._modules.keys()):
            module = self._modules[name]
            
            if isinstance(module, Tensor):
                # Move tensor and update reference
                moved = module.to(device)
                self._modules[name] = moved
                setattr(self, name, moved)
            elif hasattr(module, 'set_device'):
                # Recursively update submodules
                module.set_device(device)
            elif hasattr(module, 'device'):
                # Directly update device attribute
                module.device = device
                # Handle layers with parameters (like Dense)
                if hasattr(module, 'parameters'):
                    for param_name in ['weights', 'bias']:
                        if hasattr(module, param_name):
                            param = getattr(module, param_name)
                            moved_param = param.to(device)
                            setattr(module, param_name, moved_param)

    def set_dtype(self, dtype):
        
        self.dtype = dtype
        for param in self.parameters:
            if hasattr(param, 'astype'):
                param.data = param.data.astype(dtype)

    def state_dict(self):
        
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

        meta = state_dict.pop('_meta', {})
        self.set_device(meta.get('device', 'cpu'))
        self.set_dtype(parse_dtype(meta.get('dtype', np.float32)))

        for name, data in state_dict.items():
            module = getattr(self, name)
            if hasattr(module, 'load_state_dict'):
                module.load_state_dict(data)
            elif isinstance(module, Tensor):
                module.data = data
        self.set_dtype(self.dtype)
    def zero_grad(self):

        for param in self.parameters:
            param.zero_grad()
    def no_grad(self):

        return Tensor.no_grad()


    def __call__(self, inputs):
        return self.forward(inputs)

    def __repr__(self):
        return f"{self.__class__.__name__}(\n" + \
               "\n".join(f"  ({name}): {module}"
                        for name, module in self._modules.items()) + "\n)"
"""



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

