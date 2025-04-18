# Neural Network Framework Documentation

## Table of Contents
0. [Installation](#installation)
1. [Overview](#overview)
2. [Tensor & Automatic Differentiation](#tensor)
3. [Layers & Modules](#layers)
   - [Dense Layer](#dense)
   - [Conv2D Layer](#conv2d)
   - [MaxPool2D Layer](#maxpool2d)
   - [Flatten Layer](#flatten)
4. [Activation Functions](#activations)
   - [Tanh, ReLU, Sigmoid, Softmax, LeakyReLU, ELU](#activation-functions)
5. [Sequential Model & Building Models](#sequential)
6. [Custom Model Building](#custommodel)
   - [BaseModel Class](#basemodel)
   - [Creating Custom Models](#creating-custom-models)
   - [Handling Parameters & Submodules](#handling-parameters-submodules)
   - [Device & Dtype Management](#model-device-dtype)
   - [Serialization](#model-serialization)
7. [Loss Functions & Metrics](#losses)
   - [MSELoss, CrossEntropy, SoftmaxCrossEntropyLoss, BCELoss](#loss-functions)
   - [Accuracy Metric](#accuracy)
8. [Optimizers](#optimizers)
   - [SGD and Adam](#optimizers-details)
9. [Device & DataType Management](#device-dtype)
10. [Training Examples](#training)
11. [Transformer Architecture Implementation](#transformer)
    - [Embedding and Positional Encoding](#embedding)
    - [Multi-Head Attention](#attention)
    - [Encoder and Decoder](#encoder-decoder)
    - [Complete Transformer Model](#complete-transformer)
12. [Saving and Loading Models](#saving-loading)
13. [Conclusion & Future Extensions](#conclusion)

---


## 0. installation <a name="installation">




# Prerequisites

Before installing the required Python packages, please ensure your system meets the following prerequisites:


1.  **Python 3.12.x Installation:**
    *   **Reason:** The project environment and its dependencies (including `numpy==1.26.4`, which is compatible with Python 3.12) are specified to work and for cuda support with **Python version 3.12.x** (e.g., 3.12.0, 3.12.1, 3.12.2, etc.). Using other Python versions (like 3.11 or 3.13) may lead to compatibility issues.
    *   **Action:** Ensure you have a Python 3.12.x interpreter installed and are using it for this project (e.g., in your virtual environment).
    *   **Verification:** You can check your active Python version by running `python --version` or `python3 --version` in your terminal.

2.  **NVIDIA CUDA Toolkit 12.x Installation: optional but recommended for gpu support **
    *   **Reason:** The project depends on `cupy-cuda12x==13.3.0`. The `cuda12x` suffix indicates that this version of CuPy is specifically built for and requires **CUDA Toolkit version 12.x** (e.g., 12.0, 12.1, 12.2, 12.3, 12.4).
    *   **Action:** You must have a compatible NVIDIA driver and the CUDA Toolkit 12.x installed on your system.
    *   **Verification:** You can typically check your installed CUDA version by running `nvcc --version` in your terminal (if `nvcc` is in your PATH).

**Summary of Dependencies Driving Requirements:**


*   Project Configuration (implicit) & `numpy==1.26.4` (compatible) -> Requires Python 3.12.x
*   `cupy-cuda12x==13.3.0` -> Requires CUDA 12.x

Please install or configure CUDA and Python according to these requirements before proceeding with the package installation steps (e.g., `pip install -r requirements.txt`).

the make a project file and download this cobra to it, then import from it to another files ..
may soon we make it pip installable package.




## 1. Overview <a name="overview"></a>

This framework is a minimal yet powerful neural network library implemented using NumPy with optional support for CuPy (for GPU acceleration). It features:
- **Automatic differentiation**: Tensors track operations to enable gradient backpropagation with proper dtype handling
- **Layer modularity**: Layers such as Dense, Conv2D, and pooling layers can be composed easily
- **Activation functions**: A variety of activations (ReLU, Sigmoid, Tanh, etc.) are provided
- **Loss functions and metrics**: Compute losses like MSE, cross-entropy, and accuracy
- **Optimizers**: Standard algorithms like SGD (with momentum) and Adam are available
- **Device and dtype management**: Seamlessly move data between CPU and GPU and control numerical precision
- **Model serialization**: Save and load state via `state_dict()` and `load_state_dict()`
- **Transformer architecture**: Built-in components for creating transformer models
- **Custom model building**: Flexible API for creating complex model architectures

---

## 2. Tensor & Automatic Differentiation <a name="tensor"></a>

### The `Tensor` Class

- **Purpose:**  
  Acts as the core data structure for the framework. It wraps a NumPy (or CuPy) array, stores metadata (e.g., device, dtype), and tracks operations for automatic differentiation.
  
- **Key Properties:**
  - `data`: The underlying array.
  - `grad`: Gradient of the tensor (initialized to zeros when `requires_grad=True`).
  - `requires_grad`: Flag indicating if the tensor requires gradient computation.
  - `device`: Either `'cpu'` or `'cuda'`. When set to `'cuda'`, operations use CuPy.
  - `dtype`: Data type of the tensor (e.g., `np.float32`).
  - `xp`: Reference to the appropriate numerical library (NumPy or CuPy).

- **Core Methods:**
  - **Autograd:**
    - `backward()`: Traverses the computation graph in reverse to propagate gradients.
    - `zero_grad()`: Resets gradients to zero.
    - `no_grad()`: Context manager to disable gradient tracking.
  - **Operations:**
    - Arithmetic (`+`, `-`, `*`, `/`), matrix multiplication (`@`), element-wise operations, reshaping, slicing, and more.
  - **Device Management:**
    - `to(device)`: Moves the tensor between CPU and GPU.
  - **Type Casting:**
    - `astype(dtype)`: Returns a new tensor with the specified data type.
  - **Special Operations:**
    - `gather()`: Select values along specified dimensions using indices.
    - `where()`: Element-wise conditional selection.
    - `pad2d()`: Add padding to spatial dimensions.
    - `one_hot()`: Convert indices to one-hot encoding.

### Example Usage:
```python
# Create a tensor with gradient tracking
a = Tensor([1, 2, 3], device='cpu', dtype=np.float32, requires_grad=True)
b = Tensor([4, 5, 6], requires_grad=True)

# Perform operations
c = a + b  # Addition
d = a * b  # Element-wise multiplication
e = a @ b.reshape(3, 1)  # Matrix multiplication
f = c.mean()  # Reduction operation

# Compute gradients
f.backward()

print("Gradient of a:", a.grad.data)
print("Gradient of b:", b.grad.data)

# Device management
if has_cupy:  # Check if CuPy is available
    a_gpu = a.to('cuda')
    print("Device:", a_gpu.device)
```

### Advanced Features

- **Broadcasting**: Tensor operations support NumPy-style broadcasting with proper gradient handling.
- **Unbroadcasting**: During backpropagation, gradients are properly unbroadcast using the `unbroadcast_grad()` utility.
- **Type Consistency**: Operations maintain and enforce dtype consistency, especially during gradient calculations.
- **Numerically Stable Operations**: Implementation of stable operations like `log()`, `exp()`, and `softmax()`.

---

## 3. Layers & Modules <a name="layers"></a>

Layers inherit from `Base_Layer` and provide a standardized interface with `forward()`, `state_dict()`, and `load_state_dict()` methods.

### Dense Layer <a name="dense"></a>

- **Description:** Implements a fully connected (linear) layer.
- **Parameters:**
  - `input_size`: Number of input features.
  - `output_size`: Number of output features.
  - `name`: Optional layer name.
  - `initialization`: Weight initialization method (default: 'xavier').
  - `device`: Computing device ('cpu' or 'cuda').
  - `dtype`: Data type for parameters (default: np.float32).
- **Attributes:**
  - `weights`: Weight matrix as a Tensor.
  - `bias`: Bias vector as a Tensor.
- **Methods:**
  - `set_device(device)`: Moves layer parameters to the specified device.
  - `forward(x)`: Computes layer output (x @ weights + bias).
  - `state_dict()`: Returns a dictionary with layer parameters.
  - `load_state_dict(state_dict)`: Loads parameters from a dictionary.
- **Example:**
  ```python
  dense = Dense(128, 10, initialization='xavier', device='cpu', dtype=np.float32)
  output = dense(input_tensor)  # Equivalent to dense.forward(input_tensor)
  ```

### Conv2D Layer <a name="conv2d"></a>

- **Description:** 2D convolution layer using stride-tricks for efficient window extraction.
- **Parameters:**
  - `in_channels`: Number of input channels.
  - `out_channels`: Number of output channels.
  - `kernel_size`: Size of the convolutional kernel.
  - `stride`: Convolution stride (default: 1).
  - `padding`: Zero-padding size (default: 0).
  - `device`: Computing device (default: 'cpu').
  - `dtype`: Data type for parameters (default: np.float32).
- **Implementation Details:**
  - Uses Xavier initialization scaled by kernel dimensions.
  - Efficient implementation with `as_strided` for window extraction.
  - Reshapes data for batch matrix multiplication between windows and kernels.
- **Example:**
  ```python
  conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
  conv_out = conv(image_tensor)  # Input shape: [batch_size, in_channels, height, width]
  ```

### MaxPool2D <a name="maxpool2d"></a>

- **Description:** Performs max pooling over 2D spatial dimensions.
- **Parameters:**
  - `kernel_size`: Size of the pooling window (default: 2).
  - `stride`: Pooling stride (default: 2).
- **Implementation Details:**
  - Uses stride tricks to extract windows without data duplication.
  - Non-differentiable operation; gradients flow through the maximum values.
- **Example:**
  ```python
  pool = MaxPool2D(kernel_size=2, stride=2)
  pooled = pool(conv_out)  # Reduces spatial dimensions by factor of stride
  ```

### Flatten Layer <a name="flatten"></a>

- **Description:** Flattens multi-dimensional input to two dimensions (batch × features).
- **Implementation Details:**
  - Preserves the batch dimension.
  - Stores original shape for potential backward operations.
- **Example:**
  ```python
  flatten = Flatten()
  flat = flatten(pooled)  # Output shape: [batch_size, flattened_features]
  ```

---

## 4. Activation Functions <a name="activations"></a>

All activation functions inherit from the abstract `Activation` class and ensure device/dtype consistency.

### Activation Base Class
The abstract `Activation` class provides a common interface:
```python
class Activation(Base_Layer):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def set_gpu(self):
        self.device = 'cuda'

    def set_cpu(self):
        self.device = 'cpu'

    @abstractmethod
    def forward(self, inputs):
        # Implemented by subclasses
        pass
```

### Available Activations <a name="activation-functions"></a>

- **Tanh:**  
  Applies the hyperbolic tangent function with range [-1, 1].
  ```python
  tanh = Tanh(device='cpu')
  activated = tanh(linear_output)
  ```

- **ReLU:**  
  Implements rectified linear unit: f(x) = max(0, x).
  ```python
  relu = ReLU(device='cpu')
  activated = relu(linear_output)
  ```

- **Sigmoid:**  
  Applies the logistic function with range (0, 1).
  ```python
  sigmoid = Sigmoid(device='cpu')
  activated = sigmoid(linear_output)
  ```

- **Softmax:**  
  Normalizes outputs to a probability distribution along a specified axis.
  ```python
  softmax = Softmax(axis=-1, device='cpu')  # Typically applied to last dimension
  probabilities = softmax(logits)
  ```

- **LeakyReLU:**  
  Allows small negative values instead of zeroing them completely.
  ```python
  leaky_relu = LeakyReLU(alpha=0.01, device='cpu')  # alpha controls leak slope
  activated = leaky_relu(linear_output)
  ```

- **ELU (Exponential Linear Unit):**  
  Provides smoother activation with negative values.
  ```python
  elu = ELU(alpha=1.0, device='cpu')
  activated = elu(linear_output)
  ```

Each activation class implements the standard `forward()`, `state_dict()`, and `load_state_dict()` methods.

---

## 5. Sequential Model & Building Models <a name="sequential"></a>

The `Sequential` class enables stacking layers in a linear chain, automatically managing data flow, parameter collection, and device settings.

### Key Features:
- **Simple Layer Stacking**: Layers are executed in the order they appear in the list.
- **Parameter Management**: Collects trainable parameters from all contained layers.
- **Device Handling**: Moves all layer parameters to specified device.
- **Gradient Control**: Provides methods to control gradient tracking and reset gradients.
- **Serialization**: Supports state saving and loading.

### Example Usage:

```python
# Create a simple feedforward neural network
model = Sequential([
    Dense(784, 256, initialization='xavier'),
    ReLU(),
    Dense(256, 128),
    ReLU(),
    Dense(128, 10)
], device='cpu')

# Forward pass
predictions = model(input_tensor)

# Get trainable parameters for optimizer
optimizer = Adam(model.parameters, lr=0.001)

# Reset gradients
model.zero_grad()

# Move model to GPU if available
if has_cupy:
    model.set_device('cuda')
    
# Disable gradient tracking
with model.no_grad():
    validation_predictions = model(validation_tensor)
```

### Serialization Example:

```python
# Save model state
state = model.state_dict()
with open('model_state.pkl', 'wb') as f:
    pickle.dump(state, f)
    
# Load model state
with open('model_state.pkl', 'rb') as f:
    state = pickle.load(f)
model.load_state_dict(state)
```

---

## 6. Custom Model Building <a name="custommodel"></a>

For more complex architectures, the framework provides a `BaseModel` class for creating custom models with advanced features like skip connections, shared weights, and custom forward passes.

### BaseModel Class <a name="basemodel"></a>

The `BaseModel` class serves as the foundation for custom models:

- **Core Features:**
  - **Automatic Module Registration**: When you assign layers, tensors, or other models as attributes, they're automatically registered in the internal `_modules` dictionary.
  - **Recursive Parameter Collection**: The `parameters` property traverses all submodules to collect trainable parameters.
  - **Device & Dtype Management**: Methods to move parameters between devices and change data types.
  - **Serialization**: Support for saving and loading model state.

### Creating Custom Models <a name="creating-custom-models"></a>

To create a custom model:
1. Subclass `BaseModel`
2. Define your layers in `__init__`
3. Implement the `forward` method

#### Basic Example: Simple CNN

```python
class SimpleCNN(BaseModel):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        # Define layers (automatically registered)
        self.conv1 = Conv2D(in_channels, 16, kernel_size=3, padding=1)
        self.relu = ReLU()
        self.pool = MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = Conv2D(16, 32, kernel_size=3, padding=1)
        self.flatten = Flatten()
        self.fc = Dense(32 * 7 * 7, num_classes)  # Adjust size based on input
        
    def forward(self, x):
        # Define computation flow
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
```

#### Advanced Example: Residual Network

```python
class ResidualBlock(BaseModel):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv2D(channels, channels, kernel_size=3, padding=1)
        self.relu = ReLU()
        self.conv2 = Conv2D(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Store input for skip connection
        residual = x
        
        # Regular convolution path
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        # Add skip connection
        out = out + residual
        out = self.relu(out)
        return out

class ResNet(BaseModel):
    def __init__(self, in_channels=3, num_blocks=3, num_classes=10):
        super().__init__()
        self.conv1 = Conv2D(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.relu = ReLU()
        self.pool = MaxPool2D(kernel_size=3, stride=2)
        
        # Create residual blocks
        self.res_blocks = []
        for i in range(num_blocks):
            self.res_blocks.append(ResidualBlock(64))
            
        self.flatten = Flatten()
        self.fc = Dense(64 * 7 * 7, num_classes)  # Adjust size based on input
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x)
            
        x = self.flatten(x)
        x = self.fc(x)
        return x
```

### Handling Parameters & Submodules <a name="handling-parameters-submodules"></a>

The `BaseModel` class automatically keeps track of parameters and submodules:

- **Parameter Collection:**
  ```python
  model = SimpleCNN()
  all_trainable_params = model.parameters  # List of all trainable Tensors
  
  # Use with optimizer
  optimizer = Adam(model.parameters, lr=0.001)
  ```

- **Gradient Zeroing:**
  ```python
  model.zero_grad()  # Clears gradients from all parameters
  ```

- **No Gradient Context:**
  ```python
  with model.no_grad():
      # Operations within this block don't track gradients
      predictions = model(test_data)
  ```

### Device & Dtype Management <a name="model-device-dtype"></a>

`BaseModel` provides comprehensive device and data type management:

- **Moving to Different Device:**
  ```python
  model = SimpleCNN()
  
  # Run on GPU if available
  if has_cupy:
      model.set_device('cuda')
      
  # Move back to CPU
  model.set_device('cpu')
  ```

- **Changing Data Type:**
  ```python
  # Default is usually np.float32
  model.set_dtype(np.float32)
  
  # Switch to double precision
  model.set_dtype(np.float64)
  ```

### Serialization <a name="model-serialization"></a>

The `BaseModel` class supports saving and loading model state:

- **Saving Model State:**
  ```python
  model = SimpleCNN()
  # ... Train the model ...
  
  # Get state dictionary
  state_dict = model.state_dict()
  
  # Save using the utility function
  save_model_parameters(model, 'cnn_model.pkl')
  
  # Or manually with pickle
  with open('model.pkl', 'wb') as f:
      pickle.dump(state_dict, f)
  ```

- **Loading Model State:**
  ```python
  model = SimpleCNN()  # Create model with same architecture
  
  # Load using the utility function
  state_dict = load_state_dict_from_file('cnn_model.pkl')
  model.load_state_dict(state_dict)
  
  # Or manually with pickle
  with open('model.pkl', 'rb') as f:
      state_dict = pickle.load(f)
  model.load_state_dict(state_dict)
  ```

### Advanced Example: Siamese Network with Parameter Sharing

```python
class SiameseNetwork(BaseModel):
    def __init__(self, in_channels=1):
        super().__init__()
        # Shared encoder - same weights for both inputs
        self.encoder = Sequential([
            Conv2D(in_channels, 64, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Conv2D(64, 128, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Flatten(),
            Dense(128 * 7 * 7, 128)  # Adjust size based on input
        ])
        
        # Final classification
        self.fc = Dense(128, 1)
        self.sigmoid = Sigmoid()
        
    def forward(self, x1, x2):
        # Pass both inputs through same encoder (weight sharing)
        feat1 = self.encoder(x1)
        feat2 = self.encoder(x2)
        
        # Compute absolute difference
        distance = (feat1 - feat2).abs()
        
        # Final classification
        out = self.fc(distance)
        out = self.sigmoid(out)
        return out
    
    def __call__(self, x1, x2):
        return self.forward(x1, x2)
```

---

## 7. Loss Functions & Metrics <a name="losses"></a>

### Loss Functions <a name="loss-functions"></a>

All loss functions inherit from the base `Loss` class, which ensures consistent interfaces and numerical stability.

#### MSELoss
Mean Squared Error loss for regression tasks:
```python
mse_loss = MSELoss()
loss = mse_loss(prediction, target)  # Compute mean squared difference
```

#### CrossEntropy
Cross-entropy loss for multi-class classification:
```python
ce_loss = CrossEntropy()
# target should be one-hot encoded
loss = ce_loss(logits, one_hot_target)
```

#### SoftmaxCrossEntropyLoss
Combined softmax and cross-entropy for classification:
```python
sce_loss = SoftmaxCrossEntropyLoss()
# Works with class indices or one-hot vectors
loss = sce_loss(logits, class_indices)
```

#### BCELoss
Binary Cross Entropy for binary classification:
```python
bce_loss = BCELoss()
# Target should be in range [0, 1]
loss = bce_loss(predictions, binary_targets)
```

### Metrics <a name="accuracy"></a>

#### Accuracy
Computes classification accuracy:
```python
accuracy_metric = Accuracy()
# Works with one-hot encoded targets or class indices
acc = accuracy_metric(predictions, targets)  # Returns value between 0 and 1
```

The Accuracy metric intelligently handles different target formats:
- Class indices (1D array of integer class labels)
- One-hot encoding (2D array where each row has a single 1)

---

## 8. Optimizers <a name="optimizers"></a>

The framework includes common optimization algorithms with learning rate scheduling.

### Base Optimizer
Abstract base class with common features:
- Learning rate management
- Learning rate decay
- Device handling

### SGD Optimizer <a name="optimizers-details"></a>

Stochastic Gradient Descent with momentum:
```python
optimizer = SGD(model.parameters, 
                lr=0.01,            # Learning rate
                momentum=0.9,       # Momentum factor
                decay=0.0001)       # Learning rate decay

# In training loop
optimizer.step()         # Update parameters based on gradients
optimizer.decay_lr()     # Optionally apply learning rate decay
```

### Adam Optimizer

Adaptive Moment Estimation optimizer:
```python
optimizer = Adam(model.parameters, 
                 lr=0.001,          # Learning rate
                 beta1=0.9,         # First moment decay rate
                 beta2=0.999,       # Second moment decay rate
                 epsilon=1e-8,      # Small constant for numerical stability
                 decay=0.0)         # Learning rate decay

# In training loop
optimizer.step()         # Update parameters based on adaptive moments
```

---

## 9. Device & DataType Management <a name="device-dtype"></a>

The framework provides comprehensive device and data type management across all components.

### Device Management

- **Checking CuPy Availability:**
  ```python
  if has_cupy:
      # GPU operations available
      device = 'cuda'
  else:
      device = 'cpu'
  ```

- **Tensor Device Movement:**
  ```python
  cpu_tensor = Tensor([1, 2, 3], device='cpu')
  
  # Move to GPU if available
  if has_cupy:
      gpu_tensor = cpu_tensor.to('cuda')
      print(gpu_tensor.device)  # 'cuda'
      
      # Move back to CPU
      cpu_tensor_again = gpu_tensor.to('cpu')
  ```

- **Layer Device Settings:**
  ```python
  dense = Dense(10, 5, device='cpu')
  
  # Move to GPU
  if has_cupy:
      dense.set_device('cuda')
  ```

- **Model Device Management:**
  ```python
  model = Sequential([...]) # or BaseModel subclass
  
  # Move entire model
  if has_cupy:
      model.set_device('cuda')
  ```

### DataType Management

- **Specifying dtypes:**
  ```python
  # Create tensor with specific dtype
  tensor = Tensor([1.0, 2.0], dtype=np.float32)
  
  # Convert dtype
  double_tensor = tensor.astype(np.float64)
  ```

- **Layer dtype Settings:**
  ```python
  # Create layer with specific dtype
  dense = Dense(10, 5, dtype=np.float32)
  ```

- **Model dtype Management:**
  ```python
  model = SimpleCNN()  # BaseModel subclass
  
  # Change dtype for all parameters
  model.set_dtype(np.float64)
  ```

- **Parsing dtype Strings:**
  The framework includes a `parse_dtype` utility for converting string representations:
  ```python
  from dense import parse_dtype
  
  # Convert string to NumPy dtype
  dtype = parse_dtype("float32")  # Returns np.float32
  dtype = parse_dtype("<class 'numpy.float64'>")  # Returns np.float64
  ```

---

## 10. Training Examples <a name="training"></a>

### Basic Classification Example

Here's a complete example training a simple network on a classification task:

```python
import numpy as np
from tensor import Tensor, has_cupy
from dense import Dense
from activations import ReLU, Softmax
from sequential import Sequential
from loss import SoftmaxCrossEntropyLoss, Accuracy
from optimizer import Adam

# Set device
device = 'cuda' if has_cupy else 'cpu'

# Generate dummy data
X = np.random.randn(100, 784).astype(np.float32)  # 100 samples of 784 features (e.g., MNIST flattened)
y = np.random.randint(0, 10, size=(100,))         # 10 classes
y_one_hot = np.eye(10)[y]                         # Convert to one-hot encoding

# Convert to Tensors
X_tensor = Tensor(X, device=device)
y_tensor = Tensor(y_one_hot, device=device)

# Create model
model = Sequential([
    Dense(784, 128, device=device),
    ReLU(),
    Dense(128, 64, device=device),
    ReLU(),
    Dense(64, 10, device=device)
], device=device)

# Setup loss, optimizer, and metrics
loss_fn = SoftmaxCrossEntropyLoss()
optimizer = Adam(model.parameters, lr=0.001)
accuracy = Accuracy()

# Training loop
num_epochs = 50
batch_size = 32

for epoch in range(num_epochs):
    total_loss = 0
    total_acc = 0
    
    # Mini-batch training
    num_batches = len(X) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        # Get batch
        X_batch = X_tensor[start_idx:end_idx]
        y_batch = y_tensor[start_idx:end_idx]
        
        # Forward pass
        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)
        acc = accuracy(predictions, y_batch)
        
        # Backward pass and optimization
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.data
        total_acc += acc
    
    # Print epoch statistics
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

# Save the trained model
from manage import save_model_parameters
save_model_parameters(model, 'trained_mlp.pkl')

print("Training complete!")
```

### CNN for Image Classification

```python
import numpy as np
from tensor import Tensor, has_cupy
from custom import BaseModel
from dense import Dense
from conv import Conv2D, MaxPool2D, Flatten
from activations import ReLU
from loss import SoftmaxCrossEntropyLoss, Accuracy
from optimizer import Adam
from manage import save_model_parameters

# Set device
device = 'cuda' if has_cupy else 'cpu'

# Create a custom CNN model
class ConvNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(1, 32, kernel_size=3, padding=1, device=device)
        self.relu = ReLU(device=device)
        self.pool = MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = Conv2D(32, 64, kernel_size=3, padding=1, device=device)
        self.flatten = Flatten()
        self.fc1 = Dense(64 * 7 * 7, 128, device=device)  # For 28x28 input images
        self.fc2 = Dense(128, 10, device=device)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Generate dummy MNIST-like data
X = np.random.randn(200, 1, 28, 28).astype(np.float32)
y = np.random.randint(0, 10, size=(200,))
y_one_hot = np.eye(10)[y]

# Convert to tensors
X_tensor = Tensor(X, device=device)
y_tensor = Tensor(y_one_hot, device=device)

# Initialize model, loss, and optimizer
model = ConvNet()
loss_fn = SoftmaxCrossEntropyLoss()
optimizer = Adam(model.parameters, lr=0.001)
accuracy = Accuracy()

# Training loop
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    total_loss = 0
    total_acc = 0
    
    # Mini-batch training
    indices = np.random.permutation(len(X))
    num_batches = len(X) // batch_size
    
    for i in range(num_batches):
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        
        # Get batch
        X_batch = X_tensor[batch_indices]
        y_batch = y_tensor[batch_indices]
        
        # Forward pass
        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)
        acc = accuracy(predictions, y_batch)
        
        # Backward pass and optimization
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.data
        total_acc += acc
    
    # Print epoch statistics
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

# Save the trained model
save_model_parameters(model, 'trained_cnn.pkl')
```

---

## 11. Transformer Architecture Implementation <a name="transformer"></a>

The framework includes components specifically designed for transformer models, including embedding layers, positional encoding, multi-head attention, and encoder-decoder architecture.

### Embedding and Positional Encoding <a name="embedding"></a>

- **Embedding Layer**:
  Maps token indices to continuous vector representations.
  
  ```python
  embedding = Embedding(vocab_size=10000, d_model=512)
  embedded = embedding(token_indices)  # token_indices shape: [batch_size, seq_len]
  ```

- **Positional Encoding**:
  Adds positional information to embeddings using sinusoidal encoding.
  
  ```python
  positional_encoding = PositionalEncoding(d_model=512, max_seq_len=1000)
  embedded_with_positions = positional_encoding(embedded)
  ```

### Multi-Head Attention <a name="attention"></a>

The `MultiHeadAttention` class implements the core attention mechanism:

```python
attention = MultiHeadAttention(d_model=512, num_heads=8)
output = attention(query, key, value, mask=None)
```

Features:
- Separate projections for queries, keys, and values
- Scaled dot-product attention
- Support for attention masking
- Multi-head parallel attention

### Encoder and Decoder <a name="encoder-decoder"></a>

- **Encoder Block**:
  Processes input sequences through self-attention and feed-forward layers.
  
  ```python
  encoder = Encoder(vocab_size=10000, d_model=512, num_heads=8)
  encoder_output = encoder(src_tokens)  # src_tokens: [batch_size, src_len]
  ```

- **Decoder Block**:
  Generates output sequences using both self-attention and cross-attention to encoder output.
  
  ```python
  decoder = Decoder(vocab_size=10000, d_model=512, num_heads=8)
  decoder_output = decoder(tgt_tokens, encoder_output)  # tgt_tokens: [batch_size, tgt_len]
  ```

### Complete Transformer Model <a name="complete-transformer"></a>

The full `Transformer` class combines the encoder and decoder:

```python
transformer = Transformer(
    src_vocab_size=10000,  # Source vocabulary size
    tgt_vocab_size=10000,  # Target vocabulary size
    d_model=512,           # Model dimension
    num_heads=8            # Number of attention heads
)

# For sequence-to-sequence tasks
outputs = transformer(src_tokens, tgt_tokens)  # Shape: [batch_size, tgt_len, tgt_vocab_size]
```

### Example: Language Translation Model

```python
from tensor import Tensor
from custom import BaseModel
from loss import SoftmaxCrossEntropyLoss
from optimizer import Adam

# Create dummy token data
src_tokens = Tensor(np.random.randint(0, 1000, size=(32, 20)))  # [batch_size, src_seq_len]
tgt_tokens = Tensor(np.random.randint(0, 1000, size=(32, 22)))  # [batch_size, tgt_seq_len]
tgt_labels = Tensor(np.random.randint(0, 1000, size=(32, 22)))  # [batch_size, tgt_seq_len]

# Create transformer model
transformer = Transformer(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    d_model=256,
    num_heads=4
)

# Training setup
loss_fn = SoftmaxCrossEntropyLoss()
optimizer = Adam(transformer.parameters, lr=0.0001)

# Single training step
logits = transformer(src_tokens, tgt_tokens)
loss = loss_fn(logits, tgt_labels)
transformer.zero_grad()
loss.backward()
optimizer.step()

print(f"Training loss: {loss.data}")
```

---

## 12. Saving and Loading Models <a name="saving-loading"></a>

The framework provides utility functions for saving and loading models.

### Saving a Model

```python
from manage import save_model_parameters

# After training your model
model = SimpleCNN()
# ... Train the model ...

# Save to a file
save_model_parameters(model, 'my_model.pkl')
```

### Loading a Model

```python
from manage import load_state_dict_from_file

# Create model with the same architecture
model = SimpleCNN()

# Load parameters
state_dict = load_state_dict_from_file('my_model.pkl')
model.load_state_dict(state_dict)

# Model is ready for inference
predictions = model(test_data)
```

### What Gets Saved

The `state_dict()` method includes:
- All parameter values (weights and biases)
- Configuration information (layer shapes, device settings, dtypes)
- Metadata that helps with correct restoration

### Direct Serialization

You can also manually handle serialization:

```python
import pickle

# Save
state_dict = model.state_dict()
with open('model.pkl', 'wb') as f:
    pickle.dump(state_dict, f)

# Load
with open('model.pkl', 'rb') as f:
    state_dict = pickle.load(f)
model.load_state_dict(state_dict)
```

---

## 13. Conclusion & Future Extensions <a name="conclusion"></a>

This framework provides a powerful yet flexible foundation for deep learning research and applications:

### Key Strengths:
- **Modularity**: Easy to extend with new layers, activations, and models
- **Automatic Differentiation**: Robust gradient computation with proper type handling
- **Device/Dtype Management**: Seamless CPU/GPU switching and numerical precision control
- **PyTorch/TensorFlow-like API**: Familiar, clean interface for building models
- **Transformer Support**: First-class components for attention-based models
- **Customizability**: Hierarchical model building with `BaseModel`

### Potential Extensions:
- **Recurrent Layers**: Add LSTM and GRU implementations
- **Dataset Handling**: Create data loading and batching utilities
- **Regularization**: Add dropout, batch normalization, and weight regularization
- **Learning Rate Schedules**: Implement more sophisticated LR scheduling
- **Distributed Training**: Add support for multi-GPU and multi-node training
- **Automatic Mixed Precision**: Add support for mixed precision training
- **Graph Visualization**: Tools to visualize model architecture and computation graph
- **Quantization**: Support for reduced precision (int8, float16) operations

### Performance Considerations:
- Use CuPy when possible for GPU acceleration
- Consider batching strategy based on your hardware
- Profile your models to identify bottlenecks
- Use appropriate dtype (typically `np.float32` for most applications)

---

