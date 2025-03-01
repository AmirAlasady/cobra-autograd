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