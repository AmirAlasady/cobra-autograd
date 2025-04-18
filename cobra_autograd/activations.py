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
        condition = inputs.data > 0
        # Fixed implementation: alpha * x for negative values
        return inputs.where(condition, self.alpha * inputs)

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