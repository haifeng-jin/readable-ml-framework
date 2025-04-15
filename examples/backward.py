import numpy as np

from framework import ops
from framework.tensor import Tensor

# Create input tensors
x = Tensor.from_numpy(np.array([[2.0, 3.0]], dtype=np.float32))  # shape (1, 2)
y = Tensor.from_numpy(
    np.array([[4.0], [5.0]], dtype=np.float32)
)  # shape (2, 1)

# Perform matrix multiplication
z = ops.matmul(x, y)  # Expected: [[2*4 + 3*5]] = [[23.0]]
s = ops.sum(z)

# Trigger backward propagation
s.backward()

# Print gradients
print("x.grad:", x.grad.numpy())  # Expected: [[4.0, 5.0]]
print("y.grad:", y.grad.numpy())  # Expected: [[2.0], [3.0]]
