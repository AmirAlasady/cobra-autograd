
import numpy as np

from activations import *
from conv import Conv2D, Flatten, MaxPool2D
from custom import BaseModel
from dense import Dense
from loss import *
from optimizer import *
from sequential import Sequential
from tensor import Tensor, has_cupy
import cupy as cp # type: ignore




"""
def run_all_tests(device='cpu'):
    
    print(f"\n=== Running tests on {device.upper()} ===")

    def test_dtype_propagation():
        print("\nTest 1: Data Type Propagation")
        dtypes = [np.float16, np.float32, np.float64]
        for dtype in dtypes:
            # Test basic operations
            a = Tensor([1, 2], dtype=dtype, device=device, requires_grad=True)
            b = Tensor([3, 4], dtype=dtype, device=device)
            c = a * b + dtype(0.5)

            # Create scalar output for backward
            output = c.sum()
            output.backward()  # Now valid as output is scalar

            assert a.grad is not None
            assert a.grad.dtype == dtype, f"Gradient dtype mismatch: {a.grad.dtype} vs {dtype}"
            a.zero_grad()

            # Test mixed dtype operations
            other_dtype = np.float64 if dtype != np.float64 else np.float32
            d = Tensor([5], dtype=other_dtype, device=device)
            e = (c + d).sum()  # Create scalar for backward pass
            e.backward()

            assert a.grad.dtype == dtype, f"Mixed dtype grad failed: {a.grad.dtype} vs {dtype}"
            a.zero_grad()

    print("Data type propagation test passed!")

    def test_core_operations():
        print("\nTest 2: Core Operation Correctness")
        # Matrix multiplication
        a = Tensor([[1,2],[3,4]], dtype=np.float64, device=device, requires_grad=True)
        b = Tensor([[2,0],[1,-1]], dtype=np.float64, device=device, requires_grad=True)
        c = a @ b
        expected = np.array([[4, -2], [10, -4]], dtype=np.float64)
        assert np.allclose(c.data, expected), "Matmul forward failed"

        # Backward pass
        grad = Tensor([[1,1],[1,1]], dtype=np.float64, device=device)
        c.backward(grad)
        assert np.allclose(a.grad.data, [[2, 0], [2, 0]]), "Matmul a grad failed"
        assert np.allclose(b.grad.data, [[4, 4], [6, 6]]), "Matmul b grad failed"
        a.zero_grad(); b.zero_grad()

        # Power and division
        d = a ** 2
        e = d / Tensor(2.0, device=device)
        e.sum().backward()
        # Corrected expected gradient
        assert np.allclose(a.grad.data, [[1, 2], [3, 4]]), f"Power/div grad failed. Got:\n{a.grad.data}"
        print("Core operations test passed!")

    def test_activations():
        print("\nTest 3: Activation Functions")
        x = Tensor([-2.0, 0.0, 3.0], dtype=np.float32, device=device, requires_grad=True)

        # ReLU
        r = x.relu()
        assert np.allclose(r.data, [0, 0, 3]), "ReLU forward failed"
        r.backward(Tensor([1,1,1], device=device))
        assert np.allclose(x.grad.data, [0, 0, 1]), "ReLU grad failed"
        x.zero_grad()

        # Sigmoid
        s = x.sigmoid()
        expected = 1/(1+np.exp(-np.array([-2,0,3])))
        assert np.allclose(s.data, expected), "Sigmoid forward failed"
        s.backward(Tensor([1,1,1], device=device))
        grad_expected = expected * (1 - expected)
        assert np.allclose(x.grad.data, grad_expected), "Sigmoid grad failed"
        x.zero_grad()

        print("Activation tests passed!")

    def test_device_handling():
        if device == 'cpu' or not has_cupy:
            return

        print("\nTest 4: Cross-Device Operations")
        cpu_tensor = Tensor([1,2], dtype=np.float32, device='cpu')
        cuda_tensor = cpu_tensor.to('cuda')
        assert cuda_tensor.device == 'cuda', "Device move failed"

        # Mixed device should fail
        try:
            _ = cpu_tensor + cuda_tensor
            assert False, "Cross-device operation should fail"
        except AssertionError:
            pass

        # Round trip test
        round_trip = cuda_tensor.to('cpu')
        assert round_trip.device == 'cpu' and np.allclose(round_trip.data, [1,2])
        print("Device handling tests passed!")

    def test_edge_cases():
        print("\nTest 5: Edge Cases")
        # Division near zero
        a = Tensor([1e-8], dtype=np.float32, device=device, requires_grad=True)
        b = a / 2
        b.backward()
        assert not np.isnan(a.grad.data).any(), "NaN in gradients"

        # Large values (test different dtypes)
        # Float16 should handle exp(10) ≈ 22026 which is < 65504 (max float16)
        big_f16 = Tensor([10], dtype=np.float16, device=device)
        assert np.isfinite(big_f16.exp().data), "Float16 exp overflow"

        # Test proper overflow handling for float32/64
        if device == 'cpu':  # Cupy handles overflows differently
            big_f32 = Tensor([100], dtype=np.float32, device=device)
            assert np.isinf(big_f32.exp().data), "Float32 exp should overflow"

        print("Edge case tests passed!")

    # Run all tests
    test_dtype_propagation()
    test_core_operations()
    test_activations()
    test_device_handling()
    test_edge_cases()

# Run on CPU
run_all_tests('cpu')

# Run on CUDA if available
if has_cupy:
    run_all_tests('cuda')
else:
    print("\nSkipping CUDA tests (CuPy not available)")


def test_transpose_dtypes():
    print("\n=== Testing Transpose Dtype Consistency ===")
    xp = np

    # Test dtype preservation
    for dtype in [np.float16, np.float32, np.float64]:
        print(f"\nTesting dtype: {dtype.__name__}")
        # Test matrix transpose
        data = xp.array([[1, 2], [3, 4]], dtype=dtype)
        x = Tensor(data, dtype=dtype, requires_grad=True)
        x_t = x.T

        # Check forward pass dtype
        assert x_t.dtype == dtype, f"Forward dtype mismatch. Expected {dtype}, got {x_t.dtype}"
        assert np.array_equal(x_t.data, data.T), "Transpose values incorrect"

        # Test backward pass
        grad_data = xp.array([[0.1, 0.2], [0.3, 0.4]], dtype=dtype)
        x_t.backward(Tensor(grad_data, dtype=dtype))

        # Check gradient dtype and values
        assert x.grad.dtype == dtype, f"Gradient dtype mismatch. Expected {dtype}, got {x.grad.dtype}"
        assert np.array_equal(x.grad.data, grad_data.T), 
        
        Backward pass failed. Expected:
        [[0.1 0.3]
         [0.2 0.4]]
        Got:
        #{x.grad.data}
        .format(x=x)

    # Test mixed dtype operations
    print("\nTesting mixed dtype operations:")
    x_f32 = Tensor([[1, 2], [3, 4]], dtype=np.float32, requires_grad=True)
    x_f64 = Tensor([[1, 2], [3, 4]], dtype=np.float64, requires_grad=True)

    # Test dtype dominance
    combinations = [
        (x_f32.T + x_f64, np.float32),
        (x_f64.T + x_f32, np.float64)
    ]

    for operation, expected_dtype in combinations:
        result = operation
        assert result.dtype == expected_dtype, \
            f"Dtype dominance failed. Expected {expected_dtype}, got {result.dtype}"
        # Verify gradients exist for mixed operations
        result.backward(Tensor(xp.ones_like(result.data), dtype=expected_dtype))
        assert operation.grad.dtype == expected_dtype, \
            f"Mixed grad dtype mismatch. Expected {expected_dtype}, got {operation.a.grad.dtype}"

    print("All transpose dtype tests passed!")

# Run the test
test_transpose_dtypes()

x=Tensor(cp.asarray([2,3]),device='cuda',dtype=cp.float64)
print(x.device)
print(x.dtype)


import numpy as np

def test_mul():
    print("=== Test: Multiplication ===")
    x = Tensor([[1, 2], [3, 4]], dtype=np.float64, requires_grad=True)
    y = Tensor([[2, 0], [1, -1]], dtype=np.float64, requires_grad=True)

    z = x * y
    assert np.array_equal(z.data, np.array([[2, 0], [3, -4]], dtype=np.float64)), "Multiplication failed"

    x.grad = Tensor(np.zeros_like(x.data))
    y.grad = Tensor(np.zeros_like(y.data))
    z.backward(Tensor(np.ones_like(z.data)))
    print(z.dtype)
    print(z.grad.dtype)
    assert np.array_equal(x.grad.data, y.data), "Backward failed for x"
    assert np.array_equal(y.grad.data, x.data), "Backward failed for y"

    print("Multiplication test passed!")

def test_matmul():
    print("=== Test: Matrix Multiplication ===")
    x = Tensor([[1, 2], [3, 4]], dtype=np.float32, requires_grad=True)
    y = Tensor([[2, 0], [1, -1]], dtype=np.float64, requires_grad=True)

    z = x @ y
    expected = np.array([[4, -2], [10, -4]], dtype=np.float64)
    assert np.array_equal(z.data, expected), "Matrix multiplication failed"
    print(z.dtype)
    x.grad = Tensor(np.zeros_like(x.data), dtype=np.float64)
    y.grad = Tensor(np.zeros_like(y.data), dtype=np.float64)
    z.backward(Tensor(np.ones_like(z.data)))
    print(z.grad.dtype)
    # CORRECTED ASSERTIONS
    correct_x_grad = np.array([[2, 0], [2, 0]], dtype=np.float64)
    correct_y_grad = np.array([[4, 4], [6, 6]], dtype=np.float64)

    assert np.array_equal(x.grad.data, correct_x_grad), f"Backward failed for x. Got {x.grad.data}"
    assert np.array_equal(y.grad.data, correct_y_grad), f"Backward failed for y. Got {y.grad.data}"

    print("Matrix multiplication test passed!")

def test_pow():
    print("=== Test: Power ===")
    x = Tensor([[1, 2], [3, 4]], dtype=np.float64, requires_grad=True)
    z = x ** 2
    expected = np.array([[1, 4], [9, 16]], dtype=np.float64)
    assert np.array_equal(z.data, expected), "Power computation failed"

    x.grad = Tensor(np.zeros_like(x.data))
    z.backward(Tensor(np.ones_like(z.data)))

    expected_grad = 2 * x.data
    assert np.array_equal(x.grad.data, expected_grad), "Backward failed for power function"
    print(z.dtype)
    print(z.grad.dtype)
    print("Power test passed!")

def test_sum():
    print("=== Test: Sum ===")
    x = Tensor([[1, 2], [3, 4]], dtype=np.float64, requires_grad=True)
    z = x.sum(axis=0)

    assert np.array_equal(z.data, np.array([4, 6], dtype=np.float64)), "Sum computation failed"

    x.grad = Tensor(np.zeros_like(x.data))
    z.backward(Tensor(np.ones_like(z.data)))

    expected_grad = np.array([[1, 1], [1, 1]], dtype=np.float64)
    assert np.array_equal(x.grad.data, expected_grad), "Backward failed for sum function"
    print(z.dtype)
    print(z.grad.dtype)
    print("Sum test passed!")

# Run all tests
test_mul()
test_matmul()
test_pow()
test_sum()

def run_gpu_tests():
    if not has_cupy:
        print("CuPy not available, skipping GPU tests")
        return

    def test_mul_gpu():
        print("\n=== Test: Multiplication (GPU) ===")
        x = Tensor([[1, 2], [3, 4]], dtype=np.float64, device='cuda', requires_grad=True)
        y = Tensor([[2, 0], [1, -1]], dtype=np.float64, device='cuda', requires_grad=True)

        # Verify device and array type
        assert isinstance(x.data, cp.ndarray), "Data should be CuPy array on GPU"
        assert x.device == 'cuda', "Tensor not on GPU"

        z = x * y
        cp.testing.assert_array_equal(z.data, cp.array([[2, 0], [3, -4]], dtype=np.float64))

        # Backward pass
        z.backward(Tensor(cp.ones_like(z.data), device='cuda'))

        print(f"z dtype: {z.dtype}, device: {z.device}")
        print(f"grad dtype: {x.grad.dtype}, device: {x.grad.device}")

        cp.testing.assert_array_equal(x.grad.data, y.data)
        cp.testing.assert_array_equal(y.grad.data, x.data)
        print("Multiplication GPU test passed!")

    def test_matmul_gpu():
        print("\n=== Test: Matrix Multiplication (GPU) ===")
        x = Tensor([[1, 2], [3, 4]], dtype=np.float32, device='cuda', requires_grad=True)
        y = Tensor([[2, 0], [1, -1]], dtype=np.float64, device='cuda', requires_grad=True)

        # Should cast y to float32 before matmul
        z = x @ y
        assert z.dtype == np.float32, "Result should inherit caller's dtype"
        assert isinstance(z.data, cp.ndarray), "Result data should be on GPU"

        expected = cp.array([[4, -2], [10, -4]], dtype=np.float32)
        cp.testing.assert_allclose(z.data, expected)

        # Backward pass
        z.backward(Tensor(cp.ones_like(z.data), dtype=np.float32, device='cuda'))

        print(f"z dtype: {z.dtype}, device: {z.device}")
        print(f"x grad dtype: {x.grad.dtype}, y grad dtype: {y.grad.dtype}")

        cp.testing.assert_allclose(x.grad.data, cp.array([[2, 0], [2, 0]], dtype=np.float32))
        cp.testing.assert_allclose(y.grad.data, cp.array([[4, 4], [6, 6]], dtype=np.float64))
        print("Matrix multiplication GPU test passed!")

    def test_pow_gpu():
        print("\n=== Test: Power (GPU) ===")
        x = Tensor([[1, 2], [3, 4]], dtype=np.float64, device='cuda', requires_grad=True)
        z = x ** 2

        # Verify GPU data type
        assert isinstance(z.data, cp.ndarray), "Data should be CuPy array"

        # Check forward pass
        cp.testing.assert_array_equal(z.data, cp.array([[1, 4], [9, 16]]))

        # Backward pass
        z.backward(Tensor(cp.ones_like(z.data), device='cuda'))

        # Use approximate comparison with tolerance
        cp.testing.assert_allclose(
            x.grad.data,
            2 * x.data,
            rtol=1e-6,  # 0.0001% relative tolerance
            atol=1e-6   # Absolute tolerance of 0.000001
        )

        print(f"z dtype: {z.dtype}, grad dtype: {x.grad.dtype}")
        print("Power GPU test passed!")

    def test_sum_gpu():
        print("\n=== Test: Sum (GPU) ===")
        x = Tensor([[1, 2], [3, 4]], dtype=np.float64, device='cuda', requires_grad=True)
        z = x.sum(axis=0)

        assert z.device == 'cuda', "Sum result should stay on GPU"
        cp.testing.assert_array_equal(z.data, cp.array([4, 6], dtype=np.float64))

        z.backward(Tensor(cp.ones_like(z.data), device='cuda'))
        cp.testing.assert_array_equal(x.grad.data, cp.ones((2,2), dtype=np.float64))

        print(f"z dtype: {z.dtype}, grad device: {x.grad.device}")
        print("Sum GPU test passed!")

    # Run GPU tests
    test_mul_gpu()
    test_matmul_gpu()
    test_pow_gpu()
    test_sum_gpu()

# Execute all tests
print("\n=== CPU Tests ===")
test_mul()
test_matmul()
test_pow()
test_sum()

print("\n=== GPU Tests ===")
run_gpu_tests()


import numpy as np

def test_mul():
    print("=== Test: Multiplication ===")
    x = Tensor([[1, 2], [3, 4]], dtype=cp.float64, requires_grad=True,device='cuda')
    y = Tensor([[2, 0], [1, -1]], dtype=cp.float64, requires_grad=True,device='cuda')
    print(x.dtype)
    print(x.device)
    print(y.grad.dtype)
    z = x * y
    assert np.array_equal(z.data, cp.array([[2, 0], [3, -4]], dtype=np.float64)), "Multiplication failed"

    x.grad = Tensor(np.zeros_like(x.data),device='cuda')
    y.grad = Tensor(np.zeros_like(y.data),device='cuda')
    z.backward(Tensor(np.ones_like(z.data),device='cuda'))
    print(z.dtype)
    print(z.grad.dtype)
    assert np.array_equal(x.grad.data, y.data), "Backward failed for x"
    assert np.array_equal(y.grad.data, x.data), "Backward failed for y"

    print("Multiplication test passed!")

def test_matmul():
    print("=== Test: Matrix Multiplication ===")
    x = Tensor([[1, 2], [3, 4]], dtype=np.float32, requires_grad=True,device='cuda')
    y = Tensor([[2, 0], [1, -1]], dtype=np.float64, requires_grad=True,device='cuda')

    z = x @ y
    expected = cp.array([[4, -2], [10, -4]], dtype=np.float64)
    assert cp.array_equal(z.data, expected), "Matrix multiplication failed"
    print(z.dtype)
    x.grad = Tensor(np.zeros_like(x.data), dtype=np.float64,device='cuda')
    y.grad = Tensor(np.zeros_like(y.data), dtype=np.float64,device='cuda')
    z.backward(Tensor(np.ones_like(z.data),device='cuda'))
    print(z.grad.dtype)
    # CORRECTED ASSERTIONS
    correct_x_grad = cp.array([[2, 0], [2, 0]], dtype=np.float64)
    correct_y_grad = cp.array([[4, 4], [6, 6]], dtype=np.float64)

    assert cp.array_equal(x.grad.data, correct_x_grad), f"Backward failed for x. Got {x.grad.data}"
    assert cp.array_equal(y.grad.data, correct_y_grad), f"Backward failed for y. Got {y.grad.data}"

    print("Matrix multiplication test passed!")

def test_pow():
    print("=== Test: Power ===")
    x = Tensor([[1, 2], [3, 4]], dtype=np.float64, requires_grad=True)
    z = x ** 2
    expected = np.array([[1, 4], [9, 16]], dtype=np.float64)
    assert np.array_equal(z.data, expected), "Power computation failed"

    x.grad = Tensor(np.zeros_like(x.data))
    z.backward(Tensor(np.ones_like(z.data)))

    expected_grad = 2 * x.data
    assert np.array_equal(x.grad.data, expected_grad), "Backward failed for power function"
    print(z.dtype)
    print(z.grad.dtype)
    print("Power test passed!")

def test_sum():
    print("=== Test: Sum ===")
    x = Tensor([[1, 2], [3, 4]], dtype=np.float64, requires_grad=True)
    z = x.sum(axis=0)

    assert np.array_equal(z.data, np.array([4, 6], dtype=np.float64)), "Sum computation failed"

    x.grad = Tensor(np.zeros_like(x.data))
    z.backward(Tensor(np.ones_like(z.data)))

    expected_grad = np.array([[1, 1], [1, 1]], dtype=np.float64)
    assert np.array_equal(x.grad.data, expected_grad), "Backward failed for sum function"
    print(z.dtype)
    print(z.grad.dtype)
    print("Sum test passed!")

# Run all tests
test_mul()
test_matmul()
test_pow()
test_sum()

x = Tensor([1.0, 2.0, 3.0], requires_grad=True ,dtype=np.float64)
x2=x.to('cpu')  # set to gpu
print(x2.dtype)

# Test clip with different dtypes
x = Tensor([-2, 0.5, 3], dtype=np.float32, requires_grad=True)
min_val = Tensor(0, dtype=np.int64)       # requires_grad=False by default
max_val = Tensor(2.5, dtype=np.float64)     # requires_grad=False by default

clipped = x.clip(min_val, max_val)


print("Original dtype:", x.dtype)           # float32
print("Min_val dtype:", min_val.dtype)      # int64
print("Max_val dtype:", max_val.dtype)      # float64
print("Clipped dtype:", clipped.dtype)      # float32 (matches original)

# Verify data casting
print("\nClipped data:", clipped.data)
# Should be: [0.0, 0.5, 2.5] as float32

# Verify gradient flow
clipped.backward(Tensor([1, 1, 1]))
print("\nx.grad:", x.grad.data)  # Should be [0., 1., 0.] as float32
print("\nx.grad:", x.grad.dtype)




# Input tensors
x = Tensor([1.0, 2.0, 3.0], requires_grad=True ,dtype=np.float32)
y = Tensor([0.0, 0.0, 0.0], requires_grad=True)
condition = Tensor([True, False, True],dtype=np.float64)
print(f'x :{x.dtype}')
print(f'y :{y.dtype}')
print(f'condition :{condition.dtype}')
# Apply where
result = x.where(condition, y)  # Result: [1.0, 0.0, 3.0]
print(f'result: {result.dtype}')
# Backpropagation
result.backward(Tensor([1.0, 1.0, 1.0]))

# Gradients
print(x.grad)  # [1.0, 0.0, 1.0]  (gradient flows only where condition is True)
print(y.grad)  # [0.0, 1.0, 0.0]  (gradient flows only where condition is False)

tensor_a = Tensor([1.0], dtype=np.float32)
tensor_b = Tensor([2.0], dtype=np.float64)
result = tensor_a + tensor_b
print(result.dtype)

import numpy as np

# --- Test Case 1: mean ---
def test_mean():
    print("=== Test: mean ===")
    # Create a 1D tensor with dtype float64.
    x = Tensor([1, 2, 3, 4], dtype=np.float64, requires_grad=True)
    # Compute the mean along all elements.
    m = x.mean()

    # Expected mean = (1+2+3+4)/4 = 2.5
    print("x.data:", x.data)
    print("mean.data:", m.data)
    print("mean.dtype:", m.dtype)  # Should match x.dtype (i.e. float64)

    # Backpropagate: the derivative of the mean with respect to each element is 1/4.
    m.backward(Tensor(1.0, dtype=np.float64))
    print("x.grad.data (should be [0.25, 0.25, 0.25, 0.25]):", x.grad.data)
    print("x.grad.dtype:", x.grad.dtype)
    print()

# --- Test Case 2: __truediv__ with scalar ---
def test_truediv_scalar():
    print("=== Test: __truediv__ with scalar ===")
    # Create a tensor with dtype float64.
    x = Tensor([2, 4, 8], dtype=np.float64, requires_grad=True)
    # Divide by a scalar constant.
    y = x / 2.0

    # Expected: [1, 2, 4]
    print("x.data:", x.data)
    print("x/2.0 data:", y.data)
    print("y.dtype (should match x.dtype):", y.dtype)

    # Backpropagate: derivative d(x/2)/dx is 1/2.
    y.backward(Tensor([1, 1, 1], dtype=np.float64))
    print("x.grad.data (should be [0.5, 0.5, 0.5]):", x.grad.data)
    print("x.grad.dtype:", x.grad.dtype)
    print()

# --- Test Case 3: __truediv__ with Tensor ---
def test_truediv_tensor():
    print("=== Test: __truediv__ with Tensor ===")
    # Create a tensor with dtype float32.
    x = Tensor([2, 4, 8], dtype=np.float32, requires_grad=True)
    # Create a divisor tensor (also float64).
    divisor = Tensor([2, 2, 2], dtype=np.float64, requires_grad=False)
    y = x / divisor

    # Expected: [1, 2, 4]
    print("x.data:", x.data)
    print("divisor.data:", divisor.data)
    print("x/divisor data:", y.data)
    print("y.dtype (should match x.dtype):", y.dtype)

    # Backpropagate: derivative of (x / divisor) with respect to x is 1/divisor.
    # In this case: [1/2, 1/2, 1/2]
    y.backward(Tensor([1, 1, 1], dtype=np.float64))
    print("x.grad.data (should be [0.5, 0.5, 0.5]):", x.grad.data)
    print("x.grad.dtype:", x.grad.dtype)
    print()

# --- Test Case 4: reciprocal ---
def test_reciprocal():
    print("=== Test: reciprocal ===")
    # Create a tensor with dtype float32.
    x = Tensor([2.0, 4.0, 8.0], dtype=np.float32, requires_grad=True)
    r = x.reciprocal()

    # Expected reciprocal: [0.5, 0.25, 0.125]
    print("x.data:", x.data)
    print("reciprocal data:", r.data)
    print("reciprocal.dtype (should match x.dtype):", r.dtype)

    # Backpropagate: the derivative of 1/x is -1/x^2.
    # So expected gradients: [-1/4, -1/16, -1/64]
    r.backward(Tensor([1, 1, 1], dtype=np.float64))
    print("x.grad.data (should be approximately [-0.25, -0.0625, -0.015625]):", x.grad.data)
    print("x.grad.dtype:", x.grad.dtype)
    print()

# Run the tests
test_mean()
test_truediv_scalar()
test_truediv_tensor()
test_reciprocal()


import numpy as np

def test_max():
    print("=== Test: max ===")
    # Create a 2x2 tensor with dtype float64.
    x = Tensor([[1.0, 2.0], [3.0, 0.5]], dtype=np.float64, requires_grad=True)

    # Compute max over axis 0
    m = x.max(axis=0, keepdims=True)

    assert np.array_equal(m.data, np.array([[3.0, 2.0]], dtype=np.float64)), "Max computation failed"
    assert m.dtype == np.float64, "Max dtype mismatch"

    # Backpropagation test
    m.backward(Tensor(np.ones_like(m.data), dtype=np.float64))

    expected_grad = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)  # Grad flows to max elements
    assert np.array_equal(x.grad.data, expected_grad), "Max backward computation failed"

    print("Max function test passed!")

def test_exp():
    print("=== Test: exp ===")
    # Create a 1D tensor with float32 dtype
    x = Tensor([0.0, 1.0, 2.0], dtype=np.float32, requires_grad=True)
    y = x.exp()

    expected_data = np.exp(x.data)
    assert np.allclose(y.data, expected_data), "Exp computation failed"
    assert y.dtype == np.float32, "Exp dtype mismatch"

    # Backpropagation test
    y.backward(Tensor(np.ones_like(y.data), dtype=np.float32))

    assert np.allclose(x.grad.data, expected_data), "Exp backward computation failed"

    print("Exp function test passed!")

def test_log():
    print("=== Test: log ===")
    # Create a tensor with strictly positive values
    x = Tensor([1.0, 2.0, 4.0], dtype=np.float64, requires_grad=True)
    y = x.log()

    expected_data = np.log(x.data + x.EPSILON)
    assert np.allclose(y.data, expected_data), "Log computation failed"
    assert y.dtype == np.float64, "Log dtype mismatch"

    # Backpropagation test
    y.backward(Tensor(np.ones_like(y.data), dtype=np.float64))

    expected_grad = 1 / (x.data + x.EPSILON)
    assert np.allclose(x.grad.data, expected_grad), "Log backward computation failed"

    print("Log function test passed!")

# Run the tests
test_max()
test_exp()
test_log()


# Create a tensor with gradient tracking.
a = Tensor([1, 2, 3], device='cpu', dtype=np.float32, requires_grad=True)
b = Tensor([4, 5, 6], requires_grad=True)

# Perform a computation.
c = a + b
c.backward(Tensor(1.0))

print("Gradient of a:", a.grad)


def test_activation_dtypes():
    print("\n=== Testing Activation Dtype Consistency ===")
    xp = np

    # Test configurations (dtype, value_range)
    configs = [(np.float16, [-2, 2]),
               (np.float32, [-5, 5]),
               (np.float64, [-10, 10])]

    for dtype, val_range in configs:
        print(f"\nTesting dtype: {dtype.__name__}")
        # Create test tensor with mixed values
        data = xp.array([val_range[0], -1, 0, 1, val_range[1]], dtype=dtype)
        x = Tensor(data, dtype=dtype, requires_grad=True)

        # Forward checks for each activation
        activations = {
            'relu': x.relu(),
            'sigmoid': x.sigmoid(),
            'tanh': x.tanh(),
            'softmax': x.softmax(),
            'leaky_relu': x.leaky_relu(0.01)
        }

        # Verify output dtypes
        for name, out in activations.items():
            assert out.dtype == dtype, f"{name} output dtype mismatch: {out.dtype} vs {dtype}"
            assert out.grad.dtype == dtype, f"{name} grad dtype mismatch: {out.grad.dtype}"

        # Backward pass checks
        for name, out in activations.items():
            # Reset gradients
            x.zero_grad()

            # Create grad with same dtype
            grad_data = xp.arange(len(data), dtype=dtype) / 10
            out.backward(Tensor(grad_data, dtype=dtype))

            # Verify gradient dtype matches
            assert x.grad.dtype == dtype, (
                f"{name} gradient dtype mismatch: {x.grad.dtype} vs {dtype}")

        print(f"All dtypes maintained for {dtype.__name__}")

    print("\nAdditional Edge Case Tests:")
    # Mixed dtype operations - ADD REQUIRES_GRAD=True
    x_float32 = Tensor([-1, 0, 2], dtype=np.float32, requires_grad=True)
    x_float64 = Tensor([-1, 0, 2], dtype=np.float64, requires_grad=True)

    # Test dtype dominance in operations
    for a, b in [(x_float32, x_float64), (x_float64, x_float32)]:
        out = a.relu() + b.sigmoid()
        assert out.dtype == a.dtype, "Dtype dominance failed in mixed operations"

        # Initialize gradient before accessing
        out.grad = Tensor(xp.ones_like(out.data), dtype=out.dtype)
        assert out.grad.dtype == a.dtype, "Mixed grad dtype mismatch"

    print("All activation dtype tests passed!")


# Run the test
test_activation_dtypes()



x=Dense(2, 4, initialization='xavier',dtype=np.float64)
x2=Dense(2, 4, initialization='xavier')
s1=x.state_dict()
print('--')
print(s1)

s2=x2.state_dict()
print('--')
print(s2)
x2.load_state_dict(s1)
print('--')
print(x2.state_dict())




def test_dense_layer():
    # For reproducibility
    np.random.seed(42)

    # --- Create Dummy Data ---
    # Suppose our input has 3 features and we use a batch size of 4.
    x_data = np.random.randn(4, 3).astype(np.float32)
    print(x_data[0].shape)
    # Create a Tensor for input (no gradient needed for input)
    x = Tensor(x_data, device='cpu', requires_grad=False)
    print(x.dtype)
    # Create dummy target data for an output with 2 features
    target_data = np.random.randn(4, 2).astype(np.float32)
    target = Tensor(target_data, device='cpu', requires_grad=False)
    print(target.dtype)
    # --- Instantiate the Dense Layer ---
    # Using Xavier initialization for better convergence.
    dense = Dense(input_size=3, output_size=2,dtype=np.float64 , initialization='xavier', device='cpu')
    print('=====dence dtypes=====')
    print(dense.weights.dtype)
    print(dense.bias.dtype)
    # --- Forward Pass ---
    # The __call__ operator makes using the layer convenient.
    output = dense(x)
    print("Forward pass output:")
    print(output.data)
    print(output.data.dtype)

    # --- Loss Computation ---
    # Let's define a simple Mean Squared Error (MSE) loss.
    # Note: (output - target) works because subtraction is overloaded.
    loss = ((output - target) ** 2).sum()
    print("\nLoss:")
    print(loss.data)

    # --- Backward Pass ---
    # Compute gradients by backpropagating from the loss.
    loss.backward()
    print('--')
    print(loss.grad)
    print(dense.parameters[0].dtype)
    print(dense.parameters[0].grad.data)
    print(dense.parameters[1].grad.data)
    print(f'0 grad: {dense.zero_grad()}')
    print(dense.parameters[0].grad.data)
    print(dense.parameters[1].grad.data)

    # --- Inspect Gradients ---
    print("\nGradients for Dense Layer Parameters:")
    print("Weights gradient:")
    print(dense.weights.grad.data)
    print("Bias gradient:")
    print(dense.bias.grad.data)
    print('state dict ')
    print(dense.state_dict())
if __name__ == "__main__":
    test_dense_layer()


layer = Dense(256, 128, dtype=np.float16)  # Layer uses float16
x = np.random.randn(10, 256).astype(np.float32)  # Input is float32
output = layer(x)  # Output will be float16
print(output.dtype)


import numpy as np

# Revised XOR dataset with each input as a column vector (2x1)
inputs = [
    Tensor([[0], [0]], requires_grad=False),  # shape: (2,1)
    Tensor([[0], [1]], requires_grad=False),  # shape: (2,1)
    Tensor([[1], [0]], requires_grad=False),  # shape: (2,1)
    Tensor([[1], [1]], requires_grad=False),  # shape: (2,1)
]

# Targets remain as 1x1 tensors (scalar outputs)
targets = [
    Tensor([[0]], requires_grad=False),  # shape: (1,1)
    Tensor([[1]], requires_grad=False),
    Tensor([[1]], requires_grad=False),
    Tensor([[0]], requires_grad=False),
]

print("Input shape:", inputs[0].shape)    # Expected (2,1)
print("Target shape:", targets[0].shape)   # Expected (1,1)

# Updated XORNet using the convention z = W^T @ x + b
class XORNet:
    def __init__(self):
        # First layer: w1 is (2,2) so that w1.T is also (2,2) for our 2x1 input.
        self.w1 = Tensor(np.random.randn(2, 2) * 0.1, requires_grad=True)  # shape: (2,2)
        self.b1 = Tensor(np.zeros((2, 1)), requires_grad=True)              # shape: (2,1)

        # Second (output) layer: w2 is (2,1) so that w2.T is (1,2), matching the 2x1 hidden output.
        self.w2 = Tensor(np.random.randn(2, 1) * 0.1, requires_grad=True)  # shape: (2,1)
        self.b2 = Tensor(np.zeros((1, 1)), requires_grad=True)             # shape: (1,1)

    def forward(self, x):
        # Hidden layer: using z = W^T @ x + b
        z1 = self.w1.T @ x + self.b1    # (w1.T: (2,2)) @ (x: (2,1)) -> (2,1)
        a1 = z1.tanh()                  # Apply tanh elementwise

        # Output layer
        z2 = self.w2.T @ a1 + self.b2   # (w2.T: (1,2)) @ (a1: (2,1)) -> (1,1)
        a2 = z2.sigmoid()               # Apply sigmoid for binary classification
        return a2

    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2]

# Training parameters
model = XORNet()
learning_rate = 0.5  # Adjusted learning rate for better convergence
epochs = 2000

for epoch in range(epochs):
    total_loss = 0
    # Zero gradients for all parameters at the start of the epoch
    for p in model.parameters():
        p.zero_grad()

    for x, y in zip(inputs, targets):
        pred = model.forward(x)
        # Compute binary cross-entropy loss
        loss = - (y * pred.log() + (Tensor(1.0) - y) * (Tensor(1.0) - pred).log()).sum()
        total_loss += loss.data.item()
        loss.backward()

    # Update parameters with gradient descent
    for p in model.parameters():
        if p.grad is not None:
            assert p.data.shape == p.grad.data.shape, \
                f"Shape mismatch: {p.data.shape} vs {p.grad.data.shape}"
            p.data -= learning_rate * p.grad.data

    if epoch % 200 == 0 or epoch == epochs - 1:
        avg_loss = total_loss / len(inputs)
        print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f}")

print("\nFinal predictions:")
for x in inputs:
    pred = model.forward(x)
    print(f"Input {x.data.ravel()} -> {pred.data.item():.4f}")


import numpy as np

# Assuming you’ve already defined Tensor, Dense, and activation functions
def print_separator():
    print("=" * 50)

def test_dense_with_activation():
    np.random.seed(42)

    # Input data
    x_data = np.array([
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [1.0, -0.5, 0.0, 0.5, -1.0]
    ], dtype=np.float32)
    x = Tensor(x_data, device='cpu', requires_grad=False)

    # Target output (for loss calculation)
    target_data = np.array([
        [0.5, -0.5],
        [-0.5, 0.5]
    ], dtype=np.float32)
    target = Tensor(target_data, device='cpu', requires_grad=False)

    # Layer and activations
    dense = Dense(input_size=5, output_size=2, initialization='xavier', device='cpu')
    activations = {
        "ReLU": ReLU(),
        "Sigmoid": Sigmoid(),
        "Tanh": Tanh(),
        "Softmax": Softmax(),
        "LeakyReLU": LeakyReLU(alpha=0.1),
        "ELU": ELU(alpha=1.0)
    }

    # Iterate through all activations
    for name, activation in activations.items():
        print_separator()
        print(f"Testing Dense layer with {name} activation")

        # Forward pass
        output = dense(x)
        activated_output = activation(output)

        print(f"\nForward output after {name}:")
        print(activated_output.data)

        # Loss (Mean Squared Error)
        loss = ((activated_output - target) ** 2).mean()
        print(f"\nLoss after {name}:")
        print(loss.data)

        # Backward pass
        dense.zero_grad()
        loss.backward()

        # Print gradients
        print(f"\nGradients for Dense Layer Parameters (after {name} activation):")
        print("Weights gradient:")
        print(dense.weights.grad.data)
        print("Bias gradient:")
        print(dense.bias.grad.data)

    print_separator()
    print("All tests completed!")


if __name__ == "__main__":
    test_dense_with_activation()
import numpy as np

def tensor_close(a, b, tol=1e-5):
    
    return np.all(np.abs(a - b) < tol)

def test_activations():
    # --------------------------
    # Test Data (CPU)
    # --------------------------
    test_inputs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    batch_inputs = np.array([
        [1.0, 2.0, 3.0],
        [-1.0, -2.0, -3.0]
    ], dtype=np.float32)

    # --------------------------
    # 1. Tanh Test
    # --------------------------
    x = Tensor(test_inputs, requires_grad=True)
    tanh = Tanh()
    out = tanh(x)
    out.sum().backward()

    # Forward check
    expected = np.tanh(test_inputs)
    assert tensor_close(out.data, expected), "Tanh forward failed"

    # Gradient check (derivative: 1 - tanh^2)
    expected_grad = 1 - expected**2
    assert tensor_close(x.grad.data, expected_grad), "Tanh backward failed"

    # --------------------------
    # 2. ReLU Test
    # --------------------------
    x = Tensor(test_inputs, requires_grad=True)
    relu = ReLU()
    out = relu(x)
    out.sum().backward()

    # Forward check
    expected = np.maximum(test_inputs, 0)
    assert tensor_close(out.data, expected), "ReLU forward failed"

    # Gradient check (1 where input > 0)
    expected_grad = (test_inputs > 0).astype(np.float32)
    assert tensor_close(x.grad.data, expected_grad), "ReLU backward failed"

    # --------------------------
    # 3. Sigmoid Test
    # --------------------------
    x = Tensor(test_inputs, requires_grad=True)
    sigmoid = Sigmoid()
    out = sigmoid(x)
    out.sum().backward()

    # Forward check
    expected = 1 / (1 + np.exp(-test_inputs))
    assert tensor_close(out.data, expected), "Sigmoid forward failed"

    # Gradient check (sigmoid * (1 - sigmoid))
    expected_grad = expected * (1 - expected)
    assert tensor_close(x.grad.data, expected_grad), "Sigmoid backward failed"

    # --------------------------
    # 4. Softmax Test (Batch)
    # --------------------------
    x = Tensor(batch_inputs, requires_grad=True)
    softmax = Softmax(axis=-1)
    out = softmax(x)

    # Create dummy coefficients to avoid constant loss
    dummy_coeff = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
    loss = (out * dummy_coeff).sum()  # Non-constant loss (Tensor)
    loss.backward()

    # Forward check
    shifted = batch_inputs - np.max(batch_inputs, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    expected = exp / np.sum(exp, axis=-1, keepdims=True)
    assert tensor_close(out.data, expected), "Softmax forward failed"

    # Gradient check (using chain rule)
    sum_term = (expected * dummy_coeff.data).sum(axis=-1, keepdims=True)
    expected_grad = expected * (dummy_coeff.data - sum_term)
    assert tensor_close(x.grad.data, expected_grad), "Softmax backward failed"

    # --------------------------
    # 6. ELU Test
    # --------------------------
    alpha = 1.0
    x = Tensor(test_inputs, requires_grad=True)
    elu = ELU(alpha=alpha)
    out = elu(x)
    out.sum().backward()

    # Forward check
    expected = np.where(test_inputs > 0, test_inputs, alpha * (np.exp(test_inputs) - 1))
    assert tensor_close(out.data, expected), "ELU forward failed"

    # Gradient check (correct derivative)
    expected_grad = np.where(test_inputs > 0, 1.0, alpha * np.exp(test_inputs))
    assert tensor_close(x.grad.data, expected_grad), "ELU backward failed"

    # --------------------------
    # Edge Case Tests
    # --------------------------
    # Zero input
    x = Tensor([0.0], requires_grad=True)
    out = ReLU()(x)
    out.backward()
    assert x.grad.data[0] == 0.0, "ReLU zero grad failed"

    # Large negative (ELU stability)
    x = Tensor([-100.0], requires_grad=True)
    out = ELU()(x)
    assert np.isclose(out.data, -1.0), "ELU stability failed"

    print("All activation tests passed!")

# Run the tests
test_activations()



X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)
print(X[0].shape)
y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float32)
class XORModel:
    def __init__(self, device='cpu'):
        self.dense1 = Dense(2, 2, initialization='xavier', device=device)
        self.activation1 = Tanh(device=device)
        self.dense2 = Dense(2, 1, initialization='xavier', device=device)
        self.activation2 = Sigmoid(device=device)

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation1(x)
        x = self.dense2(x)
        return self.activation2(x)

    def parameters(self):
        return self.dense1.parameters + self.dense2.parameters

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

def train(model, X, y, epochs, lr=0.1):
    # Convert training data to tensors
    X_tensor = Tensor(X, requires_grad=False)
    y_tensor = Tensor(y, requires_grad=False)

    for epoch in range(epochs):
        # Forward pass for entire batch
        outputs = model.forward(X_tensor)

        # Compute batch loss (MSE)
        loss = ((outputs - y_tensor) ** 2).mean()

        # Backward pass
        loss.backward()

        # Update parameters (SGD)
        for param in model.parameters():
            param.data -= lr * param.grad.data

        # Zero gradients after batch update
        model.zero_grad()

        # Print progress
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.data:.6f}")

    # Final predictions
    print("\nFinal predictions:")
    with Tensor.no_grad():
        preds = model.forward(X_tensor).data
        for i in range(4):
            print(f"Input: {X[i]} => Pred: {preds[i][0]:.4f} (Target: {y[i][0]})")

# Initialize and train with batch processing
model = XORModel()
train(model, X, y, epochs=5000, lr=0.1)



# Create model using Sequential
model = Sequential([
    Dense(2, 4, initialization='xavier',dtype=np.float64),
    Tanh(),
    Dense(4, 1, initialization='xavier',dtype=np.float64),
    Sigmoid()
])

# Training data
X = Tensor([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y = Tensor([[0], [1], [1], [0]], dtype=np.float32)

# Training loop
def train(model, X, y, epochs=3000, lr=0.1):
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)

        # Compute loss
        loss = ((outputs - y) ** 2).mean()

        # Backprop
        loss.backward()

        # Update params
        for param in model.parameters:
            param.data -= lr * param.grad.data

        # Reset gradients
        model.zero_grad()

        # Print progress
        if epoch % 300 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.data:.6f}")

    # Final predictions
    print("\nFinal predictions:")
    with Tensor.no_grad():
        preds = model(X).data
        for i in range(4):
            print(f"{X.data[i]} => {preds[i][0]:.4f} (target: {y.data[i][0]})")

# Train the model
model.set_device('cpu')

train(model, X, y, epochs=3000, lr=0.1)




print(model.state_dict())
xd=model.state_dict()
print('mo1-----')
model2 = Sequential([
    Dense(2, 4, initialization='xavier'),
    Tanh(),
    Dense(4, 1, initialization='xavier'),
    Sigmoid()
])
print('mo 2-----std')
print(model2.state_dict())
model2.load_state_dict(xd)
print('mo 2-----std must =std model')
print(model2.state_dict())
print("\nFinal predictions:")
with Tensor.no_grad():
    preds = model2(X).data
    for i in range(4):
        print(f"{X.data[i]} => {preds[i][0]:.4f} (target: {y.data[i][0]})")


import numpy as np
from tensorflow.keras.datasets import mnist  # type: ignore # For easy data loading

# Load and preprocess MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize and flatten images
X_train = X_train.reshape(-1, 28*28).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 28*28).astype(np.float32) / 255.0

# Convert labels to one-hot encoding
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y].astype(np.float32)

y_train_onehot = one_hot(y_train)
y_test_onehot = one_hot(y_test)

# Create MLP model using Sequential
model = Sequential([
    Dense(784, 256, initialization='xavier'),
    ReLU(),
    Dense(256, 128, initialization='xavier'),
    ReLU(),
    Dense(128, 10, initialization='xavier'),
    Softmax(axis=-1)
])

# Training parameters
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.001

# Training loop with mini-batches
def train(model, X_train, y_train, X_test, y_test):
    for epoch in range(EPOCHS):
        # Training
        epoch_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, len(X_train), BATCH_SIZE):
            # Get batch
            X_batch = Tensor(X_train[i:i+BATCH_SIZE])
            y_batch = Tensor(y_train_onehot[i:i+BATCH_SIZE])

            # Forward
            logits = model(X_batch)
            loss = (-logits.log() * y_batch).sum(axis=1).mean()

            # Backward
            loss.backward()

            # Update
            for param in model.parameters:
                param.data -= LR * param.grad.data

            model.zero_grad()

            # Track metrics
            epoch_loss += loss.data * X_batch.shape[0]
            predictions = np.argmax(logits.data, axis=1)
            correct += np.sum(predictions == np.argmax(y_batch.data, axis=1))
            total += X_batch.shape[0]

        # Validation
        test_logits = model(Tensor(X_test))
        test_preds = np.argmax(test_logits.data, axis=1)
        test_acc = np.mean(test_preds == y_test)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {epoch_loss/total:.4f} | "
              f"Train Acc: {correct/total:.4f} | "
              f"Test Acc: {test_acc:.4f}")

# Run training
train(model, X_train, y_train_onehot, X_test, y_test)




import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
# --------------------------
# 1. Generate Spiral Dataset
# --------------------------
def generate_spiral_data(n_samples=3000, noise=0.15):

    angles = np.linspace(0, 4*np.pi, n_samples//3)

    # Class 0
    x0 = np.stack([-angles*np.cos(angles), angles*np.sin(angles)]).T * 0.4
    x0 += np.random.normal(scale=noise, size=x0.shape)

    # Class 1
    x1 = np.stack([angles*np.cos(angles), -angles*np.sin(angles)]).T * 0.4
    x1 += np.random.normal(scale=noise, size=x1.shape)

    # Class 2
    x2 = np.stack([np.cos(angles)*angles*0.4, np.zeros_like(angles)]).T
    x2 += np.random.normal(scale=noise*2, size=x2.shape)

    X = np.vstack([x0, x1, x2])
    y = np.hstack([np.zeros(n_samples//3),
                  np.ones(n_samples//3),
                  2*np.ones(n_samples//3)])

    # Shuffle and split
    permutation = np.random.permutation(len(X))
    return X[permutation], y[permutation]

# Generate and visualize data
X, y = generate_spiral_data()
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', s=5)
plt.title("Spiral Dataset (3 classes)")
plt.show()

# --------------------------
# 2. Build Complex Model
# --------------------------
model = Sequential([
    Dense(2, 64, initialization='xavier'),
    ReLU(),
    Dense(64, 32, initialization='xavier'),
    ReLU(),
    Dense(32, 16, initialization='xavier'),
    ReLU(),
    Dense(16, 3, initialization='xavier'),
    Softmax(axis=-1)
])
model.set_device('cpu')
# --------------------------
# 3. Training Loop with Advanced Features
# --------------------------
def train_spiral(model, X, y, epochs=2000, lr=0.01, batch_size=64):
    # Convert to one-hot
    y_onehot = np.eye(3)[y.astype(int)]

    # Learning rate scheduler
    def lr_scheduler(epoch, base_lr=0.01):
        return base_lr * (0.95 ** (epoch // 100))

    # Training metrics storage
    losses, accuracies = [], []

    for epoch in range(epochs):
        # Mini-batch training
        epoch_loss = 0
        correct = 0

        for i in range(0, len(X), batch_size):
            # Get batch
            X_batch = Tensor(X[i:i+batch_size])
            y_batch = Tensor(y_onehot[i:i+batch_size])

            # Forward pass
            logits = model(X_batch)
            loss = (-logits.log() * y_batch).sum(axis=1).mean()

            # L2 regularization (weight decay)
            l2_loss = 0.0001 * sum((p**2).sum() for p in model.parameters)
            total_loss = loss + l2_loss

            # Backward pass
            total_loss.backward()

            # Update with current learning rate
            current_lr = lr_scheduler(epoch, lr)
            for param in model.parameters:
                param.data -= current_lr * param.grad.data

            model.zero_grad()

            # Track metrics
            epoch_loss += loss.data
            preds = np.argmax(logits.data, axis=1)
            correct += np.sum(preds == y[i:i+batch_size])

        # Store metrics
        losses.append(epoch_loss / (len(X)/batch_size))
        accuracies.append(correct / len(X))

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {losses[-1]:.4f} | "
                  f"Acc: {accuracies[-1]:.4f} | LR: {current_lr:.5f}")

    # Plot results
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.subplot(1,2,2)
    plt.plot(accuracies)
    plt.title("Training Accuracy")
    plt.show()

    return model

# --------------------------
# 4. Train and Visualize
# --------------------------
trained_model = train_spiral(model, X, y)

# Decision boundary visualization
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:,0].min()-0.1, X[:,0].max()+0.1
    y_min, y_max = X[:,1].min()-0.1, X[:,1].max()+0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    with Tensor.no_grad():
        Z = model(np.c_[xx.ravel(), yy.ravel()]).data
    Z = np.argmax(Z, axis=1).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', s=5, edgecolors='k')
    plt.title("Learned Decision Boundaries")
    plt.show()

plot_decision_boundary(trained_model, X, y)


# Example to use MSE loss for training

# Assume pred and target are tensors returned by the model and ground truth
pred = Tensor(np.array([[0.8], [0.1], [0.9], [0.05]]), dtype=np.float32)
target = Tensor(np.array([[1.0], [0.0], [1.0], [0.0]]), dtype=np.float32)

mse_loss = MSELoss()
loss = mse_loss(pred, target)
print(f"MSE Loss: {loss.data}")

# Example to use Accuracy for evaluation
accuracy = Accuracy()
acc = accuracy(pred, target)
print(f"Accuracy: {acc}")


# Define XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR inputs
y = np.array([[0], [1], [1], [0]])  # XOR outputs

# Convert the dataset to Tensors
X = Tensor(X, dtype=np.float32)
y = Tensor(y, dtype=np.float32)

# Build the model using your predefined classes
model = Sequential([
    Dense(2, 4),  # Hidden layer with 4 neurons
    Sigmoid(),
    Dense(4, 1),  # Output layer with 1 neuron
    Sigmoid()
])

# Define the MSE loss function from the predefined class
loss_fn = MSELoss()

# Train the model
epochs = 5000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward pass
    output = model(X)

    # Compute the loss
    loss = loss_fn(output, y)
    loss.backward()

    # Update parameters
    for param in model.parameters:
        param.data -= learning_rate * param.grad.data
        param.zero_grad()
    # Print loss every 500 epochs
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}")

    # Backpropagation (manual gradient descent)


# Test the model after training
predictions = model(X).data
print("Predictions after training:", predictions.round())

# Test sum with keepdims
x = Tensor([[1, 2], [3, 4]], requires_grad=True)
sum_x = x.sum(axis=1, keepdims=True)
sum_x.backward(Tensor([[1], [1]]))
print(x.grad.data)  # Should be [[1, 1], [1, 1]]

# Test sum without keepdims
x = Tensor([[1, 2], [3, 4]], requires_grad=True)
sum_x = x.sum(axis=0)
sum_x.backward(Tensor([4, 6]))
print(x.grad.data)  # Should be [[4, 6], [4, 6]]


# Test accuracy calculation
logits = Tensor([[2.0, 1.0, 0.1],
                [0.5, 2.0, 0.3]])
targets = Tensor([0, 1])  # Class indices

acc = Accuracy()
print(acc(logits, targets))  # Should output 1.0 (both correct)

# Test with GPU tensors
logits_gpu = logits.to('cpu')
targets_gpu = targets.to('cpu')
print(acc(logits_gpu, targets_gpu))  # Should also output 1.0


# Generate spiral data (from previous example)
X, y = generate_spiral_data(n_samples=3000, noise=0.2)

# Create model
model = Sequential([
    Dense(2, 128, initialization='xavier'),
    ReLU(),
    Dense(128, 64, initialization='xavier'),
    ReLU(),
    Dense(64, 3, initialization='xavier'),
    Softmax(axis=-1)
])

# Initialize optimizer and loss
optimizer = Adam(
    params=model.parameters,
    lr=0.005,
    decay=0.001,  # 0.1% decay per epoch
    beta1=0.9,
    beta2=0.999
)
loss_fn = CrossEntropy()
accuracy = Accuracy()

# Training loop
def train(model, X, y, optimizer, epochs=500):
    y_onehot = np.eye(3)[y.astype(int)]
    batch_size = 256

    for epoch in range(epochs):
        # Mini-batch training
        epoch_loss = 0
        correct = 0

        for i in range(0, len(X), batch_size):
            # Get batch
            X_batch = Tensor(X[i:i+batch_size])
            y_batch = Tensor(y_onehot[i:i+batch_size])

            # Forward pass
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()
            model.zero_grad()

            # Track metrics
            epoch_loss += loss.data
            correct += accuracy(logits, y_batch) * X_batch.shape[0]

        # Apply learning rate decay
        optimizer.decay_lr()

        # Calculate epoch metrics
        avg_loss = epoch_loss / (len(X)/batch_size)
        acc = correct / len(X)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f} | "
                  f"Acc: {acc:.4f} | LR: {optimizer.lr:.5f}")

    # Final evaluation
    with Tensor.no_grad():
        logits = model(Tensor(X))
        final_acc = accuracy(logits, Tensor(y_onehot))
        print(f"\nFinal Accuracy: {final_acc:.4f}")

# Start training
train(model, X, y, optimizer, epochs=500)

# Test accuracy with different target formats
logits = Tensor([[2.0, 1.0], [0.5, 2.0]])
targets_class = Tensor([0, 1])          # Class indices
targets_onehot = Tensor([[1, 0], [0, 1]])  # One-hot

acc = Accuracy()
print(acc(logits, targets_class))   # Should output 1.0
print(acc(logits, targets_onehot))  # Should output 1.0

def test_tensor_operations():
    print("\n=== Testing Tensor Operations ===")

    # Basic initialization
    t1 = Tensor([1, 2, 3], requires_grad=True)
    assert t1.data.tolist() == [1, 2, 3], "Initialization failed"

    # GPU-CPU conversion
    if has_cupy:
        t_gpu = t1.to('cuda')
        assert t_gpu.device == 'cuda', "GPU conversion failed"
        t_cpu = t_gpu.to('cpu')
        assert t_cpu.device == 'cpu', "CPU conversion failed"

    # Arithmetic operations
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    c = a * b + 5
    c.backward()
    assert c.data == 11.0, "Arithmetic failed"
    assert a.grad.data == 3.0 and b.grad.data == 2.0, "Gradients failed"

    # Matrix multiplication
    m1 = Tensor([[1, 2], [3, 4]], requires_grad=True)
    m2 = Tensor([[5], [6]], requires_grad=True)
    result = m1 @ m2
    result.backward(Tensor([[1], [1]]))
    assert result.data.tolist() == [[17], [39]], "Matmul failed"
    assert m2.grad.data.tolist() == [[4], [6]], "Matmul grad failed"  # Corrected line

    print("All tensor operations tests passed!")

def test_activations():
    print("\n=== Testing Activation Functions ===")

    # ReLU
    x = Tensor([-1, 0, 2], requires_grad=True)
    relu = ReLU()
    out = relu(x)
    out.backward(Tensor([1, 1, 1]))
    assert out.data.tolist() == [0, 0, 2], "ReLU forward failed"
    assert x.grad.data.tolist() == [0, 0, 1], "ReLU grad failed"

    # Sigmoid
    x = Tensor([0], requires_grad=True)
    sig = Sigmoid()
    out = sig(x)
    out.backward()
    assert np.isclose(out.data, 0.5), "Sigmoid forward failed"
    assert np.isclose(x.grad.data, 0.25), "Sigmoid grad failed"

    # Softmax
    x = Tensor([[1, 2, 3]], requires_grad=True)
    sm = Softmax(axis=-1)
    out = sm(x)
    assert np.isclose(out.data.sum(), 1.0), "Softmax sum != 1"

    # Tanh
    x = Tensor([0], requires_grad=True)
    tanh = Tanh()
    out = tanh(x)
    out.backward()
    assert out.data == 0.0, "Tanh forward failed"
    assert x.grad.data == 1.0, "Tanh grad failed"

    print("All activation tests passed!")




def test_layers():
    print("\n=== Testing Layers ===")

    # Dense layer forward/backward
    dense = Dense(2, 3, initialization='xavier')
    x = Tensor([[1, 2]], requires_grad=True)
    out = dense(x)
    out.backward(Tensor([[1, 1, 1]]))
    assert out.shape == (1, 3), "Dense forward shape failed"
    assert dense.weights.grad is not None, "Weight grads missing"
    assert dense.bias.grad is not None, "Bias grads missing"

    # Sequential model
    model = Sequential([
        Dense(2, 4),
        ReLU(),
        Dense(4, 1),
        Sigmoid()
    ])
    out = model(Tensor([[1, 2]]))
    assert out.shape == (1, 1), "Sequential forward failed"

    # Device movement
    if has_cupy:
        model.set_device('cuda')
        assert model.layers[0].weights.device == 'cuda', "Device move failed"

    print("All layer tests passed!")




def test_losses():
    print("\n=== Testing Loss Functions ===")
    # MSE Test
    pred = Tensor([[1], [2]], requires_grad=True)
    target = Tensor([[0], [0]])
    mse = MSELoss()
    loss = mse(pred, target)
    loss.backward()

    assert loss.data == 2.5, "MSE calculation failed"
    assert np.allclose(pred.grad.data, [[1.0], [2.0]]), \
        f"MSE grad failed. Got {pred.grad.data.tolist()}, expected [[1.0], [2.0]]"

    # CrossEntropy
    logits = Tensor([[2.0, 1.0], [0.5, 2.0]], requires_grad=True)
    targets = Tensor([0, 1])
    ce = CrossEntropy()
    loss = ce(logits, targets)
    loss.backward()
    assert 0.2 < loss.data < 0.5, "CE value out of expected range"
    assert logits.grad.data.shape == (2, 2), "CE grad shape wrong"

     # BCELoss Test
    pred = Tensor([0.7, 0.3], requires_grad=True)
    target = Tensor([1, 0])
    bce = BCELoss()
    loss = bce(pred, target)

    # Forward pass check
    expected_loss = - (np.log(0.7) + np.log(0.7)) / 2  # 0.7 and 1-0.3=0.7
    assert np.isclose(loss.data, expected_loss), "BCE value incorrect"

    # Backward pass
    loss.backward()
    expected_grad = (np.array([0.7, 0.3]) - np.array([1.0, 0.0])) \
                    / (np.array([0.7*0.3, 0.3*0.7])) / 2
    expected_grad = expected_grad.astype(pred.dtype)

    assert np.allclose(pred.grad.data, expected_grad, atol=1e-4), \
        f"BCE grad failed. Expected {expected_grad}, got {pred.grad.data}"

    print("All loss tests passed!")



def test_optimizers():
    print("\n=== Testing Optimizers ===")


    # --- SGD Test with Momentum ---
    model = Sequential([Dense(2, 1)])
    initial_weights = model.parameters[0].data.copy()

    sgd = SGD(model.parameters, lr=0.1, momentum=0.9)

    # First update (velocity initialized to 0)
    for p in model.parameters:
        p.grad = Tensor(np.ones_like(p.data))
    sgd.step()

    # Velocity after first step: 0.9*0 + 0.1*1 = 0.1
    expected_velocity = 0.1
    actual_velocity = sgd.velocities[0]
    assert np.allclose(actual_velocity, expected_velocity), \
        f"SGD velocity incorrect. Expected {expected_velocity}, got {actual_velocity}"

    # Parameter update should be: -lr * velocity = -0.1 * 0.1 = -0.01
    expected_weights = initial_weights - 0.1 * 0.1
    assert np.allclose(model.parameters[0].data, expected_weights), \
        "SGD first step failed"

    # Second update (velocity accumulates momentum)
    sgd.step()
    expected_velocity = 0.19
    actual_velocity = sgd.velocities[0]
    assert np.allclose(actual_velocity, expected_velocity), \
        f"SGD velocity incorrect on second step. Expected {expected_velocity}, got {actual_velocity}"

    print("Optimizer tests passed!")

    # Adam optimizer
    model = Sequential([Dense(2, 1)])
    adam = Adam(model.parameters, lr=0.001)

    # First step
    for p in model.parameters:
        p.grad = Tensor(np.ones_like(p.data))
    adam.step()

    print("Optimizer tests passed (visual inspection needed)!")


def test_metrics():
    print("\n=== Testing Metrics ===")

    # Accuracy
    logits = Tensor([[2.0, 1.0], [0.5, 2.0], [1.0, 0.0]])
    targets = Tensor([0, 1, 1])
    acc = Accuracy()
    assert acc(logits, targets) == 2/3, "Accuracy calculation failed"

    # One-hot accuracy
    targets_oh = Tensor([[1, 0], [0, 1], [0, 1]])
    assert acc(logits, targets_oh) == 2/3, "One-hot accuracy failed"

    print("All metric tests passed!")



def test_integration():
    print("\n=== Testing Full Integration ===")

    # Generate data with explicit 2D shapes
    X = Tensor(np.random.randn(100, 3))  # (batch_size, features)
    true_weights = Tensor([[1.5], [-2.0], [0.5]])  # (features, 1)
    y = X @ true_weights + 0.3  # (100, 1)

    model = Sequential([
        Dense(3, 1, initialization='zero'),  # Input 3, Output 1
    ])

    # Debug initial state
    print("Initial weights:", model.layers[0].weights.data.T)

    opt = SGD(model.parameters, lr=0.1)
    loss_fn = MSELoss()

    for epoch in range(500):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        model.zero_grad()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
            print("Current weights:", model.layers[0].weights.data.T)
            print("Gradients:", model.layers[0].weights.grad.data.T if model.layers[0].weights.grad else None)

    final_weights = model.layers[0].weights.data.T[0]
    print("\nFinal weights:", final_weights)
    print("True weights:", true_weights.data.T[0])

    assert np.allclose(final_weights, true_weights.data.T[0], atol=0.5), \
        f"Weights not learned. Expected ~{true_weights.data.T[0]}, got {final_weights}"

    print("Integration tests passed!")



def run_all_tests():
    test_tensor_operations()
    test_activations()
    test_layers()
    test_losses()
    test_optimizers()
    test_metrics()
    test_integration()

run_all_tests()
"""



"""
class SimpleMLP(BaseModel):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.dense1 = Dense(input_size, hidden_size)
        self.relu = ReLU()
        self.dense2 = Dense(hidden_size, num_classes)
        self.softmax = Softmax()

    # No need to override forward - uses default sequential processing
def test_default_forward():
    model = SimpleMLP(784, 128, 10)
    x = Tensor(np.random.randn(32, 784), device='cpu')

    with Tensor.no_grad():
        output = model(x)
        print("Output shape:", output.shape)
        print("Sample predictions:", output.data[0][:5])

test_default_forward()



class ComplexModel(BaseModel):
    def __init__(self, input_size):
        super().__init__()
        self.feature_extractor = Dense(input_size, 128)
        self.branch1 = Dense(128, 64)
        self.branch2 = Dense(128, 64)
        self.combiner = Dense(128, 10)

    def forward(self, x):
        # Custom forward logic
        x = self.feature_extractor(x).relu()
        x1 = self.branch1(x).tanh()
        x2 = self.branch2(x).sigmoid()
        return self.combiner(x1 + x2).softmax()
def test_default_forward():
    model = SimpleMLP(784, 128, 10)
    x = Tensor(np.random.randn(32, 784), device='cpu')

    with Tensor.no_grad():
        output = model(x)
        print("Output shape:", output.shape)
        print("Sample predictions:", output.data[0][:5])

test_default_forward()



# --------------------------
# 1. Generate Spiral Dataset
# --------------------------
def generate_spiral_data(n_samples=3000, noise=0.15):
    #Generate 3-class spiral dataset
    angles = np.linspace(0, 4*np.pi, n_samples//3)

    # Class 0
    x0 = np.stack([-angles*np.cos(angles), angles*np.sin(angles)]).T * 0.4
    x0 += np.random.normal(scale=noise, size=x0.shape)

    # Class 1
    x1 = np.stack([angles*np.cos(angles), -angles*np.sin(angles)]).T * 0.4
    x1 += np.random.normal(scale=noise, size=x1.shape)

    # Class 2
    x2 = np.stack([np.cos(angles)*angles*0.4, np.zeros_like(angles)]).T
    x2 += np.random.normal(scale=noise*2, size=x2.shape)

    X = np.vstack([x0, x1, x2])
    y = np.hstack([np.zeros(n_samples//3),
                  np.ones(n_samples//3),
                  2*np.ones(n_samples//3)])

    # Shuffle and split
    permutation = np.random.permutation(len(X))
    return X[permutation], y[permutation]

# Generate and visualize data
X, y = generate_spiral_data()
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', s=5)
plt.title("Spiral Dataset (3 classes)")
plt.show()

# --------------------------
# 2. Build Complex Model
# --------------------------
class MyClassifier(BaseModel):
    def __init__(self):
        super().__init__()
        # Mixed architecture with Sequential and standalone layers
        self.feature_extractor = Sequential([
            Dense(2, 64),
            ReLU(),
            Dense(64, 32),
            ReLU(),
            Dense(32, 16),
            ReLU(),
        ])
        self.classifier = Dense(16, 3, initialization='xavier')
        self.softmax = Softmax(axis=-1)

model=MyClassifier()
model.set_device('cuda')
losscls=CrossEntropy()
optimizer=Adam(model.parameters,lr=0.01)
# --------------------------
# 3. Training Loop with Advanced Features
# --------------------------
def train_spiral(model, X, y, epochs=2000, lr=0.01, batch_size=64):
    # Convert to one-hot
    y_onehot = np.eye(3)[y.astype(int)]

    # Learning rate scheduler
    def lr_scheduler(epoch, base_lr=0.01):
        return base_lr * (0.95 ** (epoch // 100))

    # Training metrics storage
    losses, accuracies = [], []

    for epoch in range(epochs):
        # Mini-batch training
        epoch_loss = 0
        correct = 0

        for i in range(0, len(X), batch_size):
            # Get batch
            X_batch = Tensor(X[i:i+batch_size]).to('cuda')
            y_batch = Tensor(y_onehot[i:i+batch_size]).to('cuda')

            # Forward pass
            logits = model(X_batch)
            #loss = (-logits.log() * y_batch).sum(axis=1).mean()
            loss=losscls(logits,y_batch)
            # L2 regularization (weight decay)
            l2_loss = 0.0001 * sum((p**2).sum() for p in model.parameters)
            total_loss = loss + l2_loss

            # Backward pass
            total_loss.backward()

            # Update with current learning rate
            #current_lr = lr_scheduler(epoch, lr)
            #for param in model.parameters:
            #    param.data -= current_lr * param.grad.data
            optimizer.step()

            model.zero_grad()

            # Track metrics
            epoch_loss += loss.data
            preds = cp.argmax(logits.data, axis=1)
            #correct += cp.sum(preds == y[i:i+cp.asarray(batch_size)])

        # Store metrics
        losses.append(epoch_loss / (len(X)/batch_size))
        #accuracies.append(correct / len(X))

       # Print progress
       # if epoch % 100 == 0:
       #     print(f"Epoch {epoch:4d} | Loss: {losses[-1]:.4f} | "
       #           f"Acc: {accuracies[-1]:.4f} | LR: {current_lr:.5f}")

    # Plot results
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.subplot(1,2,2)
    plt.plot(accuracies)
    plt.title("Training Accuracy")
    plt.show()

    return model

# --------------------------
# 4. Train and Visualize
# --------------------------
trained_model = train_spiral(model, X, y)

# Decision boundary visualization
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:,0].min()-0.1, X[:,0].max()+0.1
    y_min, y_max = X[:,1].min()-0.1, X[:,1].max()+0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    with Tensor.no_grad():
        Z = model(np.c_[xx.ravel(), yy.ravel()]).data
    Z = np.argmax(Z, axis=1).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', s=5, edgecolors='k')
    plt.title("Learned Decision Boundaries")
    plt.show()

plot_decision_boundary(trained_model, X, y)



print('m1')
c=model.state_dict()
print(c)

model2=MyClassifier()
print('m2')
c2=model2.state_dict()
print(c2)

model2.load_state_dict(c)
print('m2 as m1')
print(model2.state_dict())

"""






"""
from tensorflow.keras.datasets import cifar10  # type: ignore # Using CIFAR-10 as our challenging dataset

# --- Define a custom model by subclassing BaseModel ---
class CustomClassifier(BaseModel):
    def __init__(self, input_dim, num_classes, device='cuda', dtype=np.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        # A three-layer MLP with mixed activations (ReLU and Tanh)
        self.fc1 = Dense(input_dim, 1024, device=device, dtype=dtype)
        self.act1 = ReLU(device=device)
        self.fc2 = Dense(1024, 512, device=device, dtype=dtype)
        self.act2 = Tanh(device=device)
        self.fc3 = Dense(512, num_classes, device=device, dtype=dtype)

    def forward(self, x):
        x = x.astype(self.dtype)  # Ensure the input is the correct type
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x

# --- Load and preprocess the CIFAR-10 dataset ---
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Flatten the images (each image: 32x32x3 becomes a vector of 3072 values) and normalize pixel values to [0,1]
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test  = x_test.reshape(x_test.shape[0], -1) / 255.0

# Flatten label arrays (they are originally of shape (N,1))
y_train = y_train.flatten()
y_test  = y_test.flatten()

# For faster testing, we use a subset of the data:
train_samples = 1000
test_samples  = 200
x_train, y_train = x_train[:train_samples], y_train[:train_samples]
x_test, y_test   = x_test[:test_samples], y_test[:test_samples]

# --- Set device to GPU ---
device = 'cuda'

# --- Instantiate the model, loss, optimizer, and accuracy metric ---
input_dim = x_train.shape[1]   # 3072 for CIFAR-10 images
num_classes = 10
model = CustomClassifier(input_dim, num_classes, device=device, dtype=np.float32)

loss_fn = CrossEntropy()  # This loss expects one-hot targets
optimizer = Adam(model.parameters, lr=0.001)
accuracy_metric = Accuracy()

# --- Training settings ---
epochs = 100
batch_size = 64
num_batches = train_samples // batch_size

print("Starting training on GPU...")
for epoch in range(epochs):
    epoch_loss = 0.0
    # Shuffle the training data each epoch
    permutation = np.random.permutation(train_samples)

    for i in range(num_batches):
        indices = permutation[i*batch_size:(i+1)*batch_size]
        x_batch = x_train[indices]
        y_batch = y_train[indices]

        x_tensor = Tensor(x_batch, device=device, dtype=np.float32)
        # Ensure target Tensor is float32 so one_hot produces float values
        y_tensor = Tensor(y_batch, device=device, dtype=np.float32).one_hot(num_classes)

        # Forward pass
        logits = model(x_tensor)

        # Compute loss (CrossEntropy loss applies softmax internally)
        loss = loss_fn(logits, y_tensor)

        # Backward pass to compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Zero out gradients for the next iteration
        model.zero_grad()

        # Retrieve loss value (for cupy arrays, use .get() to move data to CPU)
        loss_value = loss.data.get().item() if device == 'cuda' else loss.data.item()
        epoch_loss += loss_value

    avg_loss = epoch_loss / num_batches

    # Evaluate training accuracy on the whole training set
    x_train_tensor = Tensor(x_train, device=device, dtype=np.float32)
    train_logits = model(x_train_tensor)
    train_pred = train_logits.argmax(axis=-1)
    # Get prediction data as a NumPy array (use .get() if on GPU)
    pred_np = train_pred.data.get() if device == 'cuda' else train_pred.data
    train_accuracy = np.mean(pred_np == y_train)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Training Accuracy: {train_accuracy:.4f}")

# --- Evaluate on test set ---
x_test_tensor = Tensor(x_test, device=device, dtype=np.float32)
test_logits = model(x_test_tensor)
test_pred = test_logits.argmax(axis=-1)
test_pred_np = test_pred.data.get() if device == 'cuda' else test_pred.data
test_accuracy = np.mean(test_pred_np == y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
"""





"""
import numpy as np
from tensorflow.keras.datasets import cifar10 # type: ignore

# --- Define a Residual Block using Dense layers ---
class ResidualBlock(Base_Layer):
    def __init__(self, size, device='cpu', dtype=np.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.fc1 = Dense(size, size, device=device, dtype=dtype)
        self.act = ReLU(device=device)
        self.fc2 = Dense(size, size, device=device, dtype=dtype)

    def forward(self, x):
        # Save input for the skip connection
        residual = x
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        return out + residual

    def __call__(self, x):
        return self.forward(x)

    def state_dict(self):
        return {
            "fc1": self.fc1.state_dict(),
            "fc2": self.fc2.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.fc1.load_state_dict(state_dict["fc1"])
        self.fc2.load_state_dict(state_dict["fc2"])

# --- Define a ResNet-like Model using Residual Blocks ---
class ResNetMLP(BaseModel):
    def __init__(self, input_dim, num_classes, num_blocks=3, hidden_size=512, device='cpu', dtype=np.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        # Initial projection to hidden dimension
        self.fc_in = Dense(input_dim, hidden_size, device=device, dtype=dtype)
        # Stack several residual blocks
        self.res_blocks = [ResidualBlock(hidden_size, device=device, dtype=dtype) for _ in range(num_blocks)]
        # Final classification layer
        self.fc_out = Dense(hidden_size, num_classes, device=device, dtype=dtype)

    def forward(self, x):
        # Ensure input tensor is of the proper dtype
        if not isinstance(x, Tensor):
            x = Tensor(x, device=self.device, dtype=self.dtype)
        else:
            if x.dtype != self.dtype:
                x = x.astype(self.dtype)
        x = self.fc_in(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.fc_out(x)
        return x

    def state_dict(self):
        state = {"fc_in": self.fc_in.state_dict(),
                 "fc_out": self.fc_out.state_dict()}
        for i, block in enumerate(self.res_blocks):
            state[f"block_{i}"] = block.state_dict()
        return state

    def load_state_dict(self, state_dict):
        self.fc_in.load_state_dict(state_dict["fc_in"])
        self.fc_out.load_state_dict(state_dict["fc_out"])
        for i, block in enumerate(self.res_blocks):
            block.load_state_dict(state_dict[f"block_{i}"])

    def zero_grad(self):
        self.fc_in.zero_grad()
        self.fc_out.zero_grad()
        for block in self.res_blocks:
            block.fc1.zero_grad()
            block.fc2.zero_grad()

# --- Load and preprocess CIFAR-10 on CPU ---
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Flatten images (32x32x3 -> 3072) and normalize to [0,1]
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test  = x_test.reshape(x_test.shape[0], -1) / 255.0

# Flatten label arrays (they are of shape (N,1))
y_train = y_train.flatten()
y_test  = y_test.flatten()

# For quick testing, use a subset of data:
train_samples = 1000
test_samples  = 200
x_train, y_train = x_train[:train_samples], y_train[:train_samples]
x_test, y_test   = x_test[:test_samples], y_test[:test_samples]

# --- Set device to CPU ---
device = 'cpu'

# --- Instantiate the ResNet-like model, loss, optimizer, and metric ---
input_dim = x_train.shape[1]  # 3072 for CIFAR-10
num_classes = 10
model = ResNetMLP(input_dim, num_classes, num_blocks=3, hidden_size=512, device=device, dtype=np.float32)

loss_fn = CrossEntropy()  # CrossEntropy loss (applies softmax internally)
optimizer = Adam(model.parameters, lr=0.001)
accuracy_metric = Accuracy()

# --- Training settings ---
epochs = 5000
batch_size = 64
num_batches = train_samples // batch_size

print("Starting training on CPU with ResNet model...")
for epoch in range(epochs):
    epoch_loss = 0.0
    permutation = np.random.permutation(train_samples)
    for i in range(num_batches):
        indices = permutation[i*batch_size:(i+1)*batch_size]
        x_batch = x_train[indices]
        y_batch = y_train[indices]

        # Create Tensors on CPU; note we use float32 for targets so that one_hot yields floats
        x_tensor = Tensor(x_batch, device=device, dtype=np.float32)
        y_tensor = Tensor(y_batch, device=device, dtype=np.float32).one_hot(num_classes)

        # Forward pass
        logits = model(x_tensor)

        # Compute loss
        loss = loss_fn(logits, y_tensor)

        # Backward pass to compute gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        # Zero gradients for the next iteration
        model.zero_grad()

        loss_value = loss.data.item()
        epoch_loss += loss_value

    avg_loss = epoch_loss / num_batches

    # Evaluate training accuracy on the full training set
    x_train_tensor = Tensor(x_train, device=device, dtype=np.float32)
    train_logits = model(x_train_tensor)
    train_pred = train_logits.argmax(axis=-1)
    train_pred_np = train_pred.data  # already on CPU
    train_accuracy = np.mean(train_pred_np == y_train)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Training Accuracy: {train_accuracy:.4f}")

# --- Evaluate on test set ---
x_test_tensor = Tensor(x_test, device=device, dtype=np.float32)
test_logits = model(x_test_tensor)
test_pred = test_logits.argmax(axis=-1)
test_pred_np = test_pred.data
test_accuracy = np.mean(test_pred_np == y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
import numpy as np


class SmallCNN(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        # Input: (batch, 3, 32, 32)
        self.conv1 = Conv2D(3, 16, 3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2)  # Output: (batch, 16, 16, 16)

        self.conv2 = Conv2D(16, 32, 3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2)  # Output: (batch, 32, 8, 8)

        self.flatten = Flatten()    # Output: (batch, 32*8*8)
        self.fc1 = Dense(32*8*8, 128)
        self.relu3 = ReLU()
        self.fc2 = Dense(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)

def generate_synthetic_data(num_samples=1000, img_size=32):
    # Generate random images and labels
    X = np.random.rand(num_samples, 3, img_size, img_size).astype(np.float32)
    y = np.random.randint(0, 10, size=num_samples)
    y = np.eye(10)[y]  # One-hot encode
    return X, y

def test_cnn_cpu():
    # Generate synthetic dataset
    X_train, y_train = generate_synthetic_data()
    X_test, y_test = generate_synthetic_data(200)

    # Convert to Tensors
    X_train = Tensor(X_train, device='cpu', dtype=np.float32)
    y_train = Tensor(y_train, device='cpu', dtype=np.float32)
    X_test = Tensor(X_test, device='cpu', dtype=np.float32)
    y_test = Tensor(y_test, device='cpu', dtype=np.float32)

    # Initialize model
    model = SmallCNN()
    criterion = SoftmaxCrossEntropyLoss()
    optimizer = Adam(model.parameters, lr=0.001)
    accuracy = Accuracy()

    # Training loop
    batch_size = 32
    epochs = 5
    # Training loop
    batch_size = 32
    epochs = 5
    num_samples = X_train.data.shape[0]  # Get from numpy array

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, num_samples, batch_size):
            end = i + batch_size

            # Get batch from numpy arrays
            inputs = Tensor(X_train.data[i:end], device='cpu')
            targets = Tensor(y_train.data[i:end], device='cpu')

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.data.item()
            preds = outputs.argmax(axis=1)
            correct += (preds.data == targets.argmax(axis=1).data).sum()
            total += inputs.data.shape[0]

        # Calculate average loss using NUM_SAMPLES from array shape
        avg_loss = epoch_loss / (num_samples / batch_size)

        # Validation
        with model.no_grad():
            test_outputs = model(X_test)
            test_acc = accuracy(test_outputs, y_test)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_loss:.4f}")  # Directly use calculated value
        print(f"Train Acc: {correct/total:.4f}")
        print(f"Test Acc: {test_acc:.4f}\n")

        # Final check
        assert epoch_loss > 0, "Model failed to train"
        print("CNN CPU Test Passed!")

if __name__ == "__main__":
    test_cnn_cpu()


def test_state_dict():
    # Initialize components
    model = SmallCNN()
    x = Tensor(np.random.randn(5, 3, 32, 32), device='cpu')  # Batch of 5 samples

    # Test Conv2D layer
    conv = Conv2D(3, 16, 3, padding=1)
    conv_state = conv.state_dict()
    assert 'kernels' in conv_state, "Conv2D kernels not in state dict"
    assert 'bias' in conv_state, "Conv2D bias not in state dict"
    assert 'config' in conv_state, "Conv2D config missing"

    # Test Dense layer
    dense = Dense(128, 10)
    dense_state = dense.state_dict()
    assert 'weights' in dense_state, "Dense weights missing"
    assert 'bias' in dense_state, "Dense bias missing"

    # Full model round-trip test
    original_output = model(x)

    # Save full state
    original_state = model.state_dict()

    # Modify model parameters
    for param in model.parameters:
        param.data += 0.1  # Corrupt parameters

    # Load original state
    model.load_state_dict(original_state)

    # Verify parameter restoration
    for (name, param), (orig_name, orig_param) in zip(model.state_dict().items(), original_state.items()):
        if 'data' in param:
            assert np.allclose(param['data'], orig_param['data']), f"{name} data mismatch"

    # Verify output consistency
    restored_output = model(x)
    assert np.allclose(original_output.data, restored_output.data, atol=1e-6), "Output mismatch after reload"

    # Test partial loading
    partial_state = {
        'conv1': model.conv1.state_dict(),
        'fc2': model.fc2.state_dict()
    }
    model.load_state_dict(partial_state)

    # Verify mixed state
    assert np.array_equal(model.conv1.kernels.data, original_state['conv1']['kernels'])
    assert np.array_equal(model.fc2.weights.data, original_state['fc2']['weights'])

    print("All state dict tests passed!")

# Integrate with existing test
def test_cnn_cpu():
    # ... previous test code ...

    # Add state dict test
    test_state_dict()

if __name__ == "__main__":
    test_cnn_cpu()
"""

import numpy as np
from tensorflow.keras.datasets import mnist

# -----------------------------
# Data Preparation
# -----------------------------
# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values and reshape to (batch, channels, height, width)
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

# -----------------------------
# Define the CNN Model
# -----------------------------
class CNN(BaseModel):
    def __init__(self, device='cpu', dtype=np.float32):
        super().__init__()
        # First convolution block
        self.conv1 = Conv2D(in_channels=1, out_channels=33, kernel_size=3,
                            stride=1, padding=1, device=device, dtype=dtype)
        self.relu1 = ReLU(device=device)
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)

        # Second convolution block
        self.conv2 = Conv2D(in_channels=33, out_channels=16, kernel_size=3,
                            stride=1, padding=1, device=device, dtype=dtype)
        self.relu2 = ReLU(device=device)
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)

        # Flatten and fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(input_size=16 * 7 * 7, output_size=128, device=device, dtype=dtype)
        self.relu3 = ReLU(device=device)
        self.fc2 = Dense(input_size=128, output_size=10, device=device, dtype=dtype)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# -----------------------------
# Instantiate Model, Loss, Optimizer, and Metric
# -----------------------------
device = 'cuda'
model = CNN(device=device, dtype=np.float32)
loss_fn = SoftmaxCrossEntropyLoss()
optimizer = Adam(model.parameters, lr=0.001)
accuracy_metric = Accuracy()

# -----------------------------
# Training Loop (Optimized)
# -----------------------------
num_epochs = 5
batch_size = 64
num_train = X_train.shape[0]
num_batches = int(np.ceil(num_train / batch_size))

for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_acc = 0.0

    # Shuffle training data
    perm = np.random.permutation(num_train)
    X_train, y_train = X_train[perm], y_train[perm]

    for i in range(0, num_train, batch_size):
        x_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Convert to Tensors
        x_tensor = Tensor(x_batch, device=device, dtype=np.float32, requires_grad=True)
        y_tensor = Tensor(y_batch, device=device, dtype=np.int64, requires_grad=False)

        # Forward pass
        preds = model(x_tensor)
        loss = loss_fn(preds, y_tensor)
        epoch_loss += float(loss.data)

        # Backward pass
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute batch accuracy
        batch_acc = accuracy_metric(preds, y_tensor)
        epoch_acc += batch_acc

    # Compute epoch metrics
    epoch_loss /= num_batches
    epoch_acc /= num_batches
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# -----------------------------
# Evaluation on Test Set
# -----------------------------
num_test = X_test.shape[0]
test_acc = 0.0
num_test_batches = int(np.ceil(num_test / batch_size))

for i in range(0, num_test, batch_size):
    x_batch = X_test[i:i+batch_size]
    y_batch = y_test[i:i+batch_size]

    x_tensor = Tensor(x_batch, device=device, dtype=np.float32, requires_grad=False)
    y_tensor = Tensor(y_batch, device=device, dtype=np.int64, requires_grad=False)

    preds = model(x_tensor)
    test_acc += accuracy_metric(preds, y_tensor)

test_acc /= num_test_batches
print(f"Test Accuracy: {test_acc:.4f}")