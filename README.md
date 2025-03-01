Below is the complete, in‐depth documentation for your neural network framework. This guide covers all aspects—from the low‐level tensor operations and automatic differentiation to layers, activations, model building, loss functions, optimizers, device/dtype management, and even a full training example.

---

# Neural Network Framework Documentation

## Table of Contents
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
6. [Loss Functions & Metrics](#losses)
   - [MSELoss, CrossEntropy, SoftmaxCrossEntropyLoss, BCELoss](#loss-functions)
   - [Accuracy Metric](#accuracy)
7. [Optimizers](#optimizers)
   - [SGD and Adam](#optimizers-details)
8. [Device & DataType Management](#device-dtype)
9. [Training Example](#training)
10. [Conclusion & Future Extensions](#conclusion)

---

## 1. Overview <a name="overview"></a>

This framework is a minimal yet powerful neural network library implemented using NumPy with optional support for CuPy (for GPU acceleration). It features:
- **Automatic differentiation**: Tensors track operations to enable gradient backpropagation.
- **Layer modularity**: Layers such as Dense, Conv2D, and pooling layers can be composed easily.
- **Activation functions**: A variety of activations (ReLU, Sigmoid, Tanh, etc.) are provided.
- **Loss functions and metrics**: Compute losses like MSE, cross-entropy, and accuracy.
- **Optimizers**: Standard algorithms like SGD (with momentum) and Adam are available.
- **Device and dtype management**: Seamlessly move data between CPU and GPU and control numerical precision.
- **Model serialization**: Save and load state via `state_dict()` and `load_state_dict()`.

---

## 2. Tensor & Automatic Differentiation <a name="tensor"></a>

### The `Tensor` Class

- **Purpose:**  
  Acts as the core data structure for your NN framework. It wraps a NumPy (or CuPy) array, stores metadata (e.g., device, dtype), and tracks operations for automatic differentiation.
  
- **Key Properties:**
  - `data`: The underlying array.
  - `grad`: Gradient of the tensor (initialized to zeros when `requires_grad=True`).
  - `requires_grad`: Flag indicating if the tensor requires gradient computation.
  - `device`: Either `'cpu'` or `'cuda'`. When set to `'cuda'`, operations use CuPy.
  - `dtype`: Data type of the tensor (e.g., `np.float32`).

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

### Example Usage:
```python
# Create a tensor with gradient tracking.
a = Tensor([1, 2, 3], device='cpu', dtype=np.float32, requires_grad=True)
b = Tensor([4, 5, 6], requires_grad=True)

# Perform a computation.
c = a + b
c.backward(Tensor(1.0))

print("Gradient of a:", a.grad)
```

---

## 3. Layers & Modules <a name="layers"></a>

Layers in this framework inherit from a base layer (implicitly managed via `Base_Layer` or auto-registration in `BaseModel`). They encapsulate parameters, forward passes, and gradient tracking.

### Dense Layer <a name="dense"></a>

- **Description:** Implements a fully connected (linear) layer.
- **Parameters:**
  - `input_size`, `output_size`
  - `initialization`: Supports methods like `'xavier'`.
  - `device` and `dtype`: Ensure consistent parameter storage.
- **Forward Pass:**  
  Computes `x @ weights + bias`.
- **Example:**
  ```python
  dense = Dense(128, 10, initialization='xavier', device='cpu', dtype=np.float32)
  output = dense(input_tensor)
  ```

### Conv2D Layer <a name="conv2d"></a>

- **Description:** 2D convolution layer using stride-tricks for efficient window extraction.
- **Parameters:**
  - `in_channels`, `out_channels`, `kernel_size`
  - `stride`, `padding`, plus `device` and `dtype`.
- **Forward Pass:**  
  Extracts sliding windows from the input and performs matrix multiplication with the kernel weights.
- **Example:**
  ```python
  conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
  conv_out = conv(image_tensor)
  ```

### MaxPool2D <a name="maxpool2d"></a>

- **Description:** Performs max pooling over 2D windows.
- **Parameters:**
  - `kernel_size`, `stride`
- **Forward Pass:**  
  Uses stride tricks to extract windows and then applies max reduction.
- **Example:**
  ```python
  pool = MaxPool2D(kernel_size=2, stride=2)
  pooled = pool(conv_out)
  ```

### Flatten Layer <a name="flatten"></a>

- **Description:** Flattens multi-dimensional input to two dimensions, preserving the batch size.
- **Usage:**
  ```python
  flatten = Flatten()
  flat = flatten(pooled)
  ```

---

## 4. Activation Functions <a name="activations"></a>

All activation functions inherit from the abstract `Activation` class and ensure that inputs are cast to the appropriate device and dtype.

### Available Activations:
- **Tanh:**  
  Applies the hyperbolic tangent function.
  ```python
  tanh = Tanh(device='cpu')
  activated = tanh(linear_output)
  ```
- **ReLU:**  
  Implements the rectified linear unit.
  ```python
  relu = ReLU(device='cpu')
  activated = relu(linear_output)
  ```
- **Sigmoid:**  
  Uses the logistic function.
- **Softmax:**  
  Computes the softmax over a specified axis.
- **LeakyReLU:**  
  Similar to ReLU but allows a small gradient when inputs are negative.
- **ELU:**  
  Exponential Linear Unit that smooths out negative inputs.

Each activation class provides `forward()`, `state_dict()`, and `load_state_dict()` methods for consistency.

---

## 5. Sequential Model & Building Models <a name="sequential"></a>

The `Sequential` class allows you to stack layers sequentially, automatically managing the propagation of data through layers, parameters collection, and device settings.

- **Key Features:**
  - **Parameter Aggregation:**  
    Collects trainable parameters from all layers.
  - **Device Management:**  
    Provides `set_device()` to move the entire model.
  - **Gradient Handling:**  
    Implements `zero_grad()` to reset gradients across layers.
  - **Serialization:**  
    Supports saving (`state_dict()`) and loading (`load_state_dict()`) model parameters.
- **Usage Example:**
  ```python
  model = Sequential([
      Dense(784, 128, initialization='xavier', device='cpu', dtype=np.float32),
      ReLU(),
      Dense(128, 10, initialization='xavier', device='cpu', dtype=np.float32)
  ], device='cpu')
  
  output = model(input_tensor)
  ```

---

## 6. Loss Functions & Metrics <a name="losses"></a>

### Loss Functions <a name="loss-functions"></a>

These functions compute the loss and ensure numerical stability.

- **MSELoss:**  
  Computes mean squared error.
  ```python
  loss_fn = MSELoss()
  loss = loss_fn(prediction, target)
  ```
- **CrossEntropy:**  
  Uses a log-sum-exp trick for stability.
- **SoftmaxCrossEntropyLoss:**  
  Combines softmax with cross-entropy, including gathering of the correct class probabilities.
- **BCELoss:**  
  Implements binary cross entropy with clipping to avoid logarithm of zero.

### Accuracy Metric <a name="accuracy"></a>

- **Description:**  
  Computes the accuracy score for classification tasks. It handles both one-hot encoded targets and class indices.
- **Usage:**
  ```python
  accuracy_metric = Accuracy()
  acc = accuracy_metric(prediction, target)
  ```

---

## 7. Optimizers <a name="optimizers"></a>

### Base Optimizer

- **Features:**  
  Provides a structure for updating parameters and includes support for learning rate decay.

### SGD Optimizer <a name="optimizers-details"></a>

- **Description:**  
  Implements momentum-based SGD.
- **Parameters:**  
  `lr`, `momentum`, and optional learning rate `decay`.
- **Usage:**
  ```python
  optimizer = SGD(model.parameters, lr=0.01, momentum=0.9)
  optimizer.step()  # Update parameters after backward pass.
  ```

### Adam Optimizer

- **Description:**  
  Implements the Adam optimizer with bias correction.
- **Parameters:**  
  `lr`, `beta1`, `beta2`, `epsilon`, and learning rate decay.
- **Usage:**
  ```python
  optimizer = Adam(model.parameters, lr=0.001)
  optimizer.step()  # Updates parameters based on adaptive moments.
  ```

---

## 8. Device & DataType Management <a name="device-dtype"></a>

### Device Management

- **Tensor Movement:**  
  Use the `to(device)` method on tensors to shift between CPU and GPU.
- **Layer and Model Device Setting:**  
  Both individual layers (via `set_device()`) and the entire model (via the `Sequential` or `BaseModel` classes) can be moved between devices.
  
### DataType Management

- **Specifying dtypes:**  
  Layers and tensors accept a `dtype` parameter (e.g., `np.float32`).  
- **Parsing Dtype Strings:**  
  The helper function `parse_dtype()` converts string representations (such as `"float32"`) into actual NumPy dtype objects.
  
### Example:
```python
# Create a tensor on GPU (if CuPy is installed) and then move it back to CPU.
a = Tensor([1, 2, 3], device='cuda', dtype=np.float32)
a_cpu = a.to('cpu')
```

---

## 9. Training Example <a name="training"></a>

Below is a complete training loop for a simple regression model using a Sequential model, MSE loss, and the SGD optimizer:

```python
import numpy as np

# Generate dummy data (e.g., for regression)
X = np.random.randn(100, 10)  # 100 samples, 10 features
y = np.random.randn(100, 1)   # 100 target values

# Convert to Tensors
X_tensor = Tensor(X, device='cpu', dtype=np.float32, requires_grad=False)
y_tensor = Tensor(y, device='cpu', dtype=np.float32, requires_grad=False)

# Build a simple model: 10 -> 50 -> 1
model = Sequential([
    Dense(10, 50, initialization='xavier', device='cpu', dtype=np.float32),
    ReLU(),
    Dense(50, 1, initialization='xavier', device='cpu', dtype=np.float32)
], device='cpu')

# Define loss and optimizer
loss_fn = MSELoss()
optimizer = SGD(model.parameters, lr=0.01, momentum=0.9)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    predictions = model(X_tensor)
    loss = loss_fn(predictions, y_tensor)
    
    # Zero gradients, backward pass, and update parameters
    model.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.decay_lr()  # Apply learning rate decay if needed

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

print("Training complete!")
```

---

## 10. Conclusion & Future Extensions <a name="conclusion"></a>

This comprehensive framework provides a robust, modular, and flexible basis for building neural networks:
- **Automatic Differentiation:** Seamlessly compute gradients for backpropagation.
- **Layer & Model Modularity:** Easily construct complex models with Dense, Conv2D, and pooling layers.
- **Rich Activation & Loss Functions:** Choose from a variety of activations and losses to suit your task.
- **Optimizers & Metrics:** Use SGD or Adam to train models and compute accuracy for evaluation.
- **Device & Dtype Flexibility:** Efficiently switch between CPU and GPU and control numerical precision.
- **Model Serialization:** Save and load model state with ease.

**Potential Extensions:**
- Integrate additional layer types (e.g., recurrent or attention-based layers).
- Enhance debugging and visualization tools for the computation graph.
- Implement distributed training mechanisms.
- Expand data augmentation and pre-processing pipelines.

This documentation should serve as a complete guide to understanding, utilizing, and extending your neural network framework. Enjoy building and training your models with this legendary toolkit!

---

Feel free to refer to any section of this guide as you develop and experiment with your neural network projects. Happy coding!