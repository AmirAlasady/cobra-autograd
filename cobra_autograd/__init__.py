"""
Cobra Autograd - A lightweight automatic differentiation library
"""

# Core components
from .tensor import Tensor
from .loss import *
from .optimizer import *
from .activations import *
from .dense import *     
from .conv import *
from .sequential import Sequential

# Version
__version__ = "0.1.0"

# Public API
__all__ = [
    'Tensor',
    'Sequential',
    'loss',
    'optimizer',
    'activations',
    'conv',
    'dence'  # Or 'dense' if you fix the typo
]