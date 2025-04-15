# Readable ML Framework

A machine learning framework with readable source code.

Machine learning frameworks can be intimidating. Their codebases are often
massive and complex, making it nearly impossible to simply read through the
source code to understand what's happening when you build a model with them.

Fortunately, there's the Readable ML Framework, which contains only about 800
lines of actual code (not counting comments) written in Python and C++. The
code is thoroughly documented, bringing the total to around 2,000 lines.

The features of the library are just enough to implement a simple neural
network for a basic classification problem without any unnecessary extras. By
reading through it, you can easily grasp the fundamentals of how an ML
framework works.

Here is a basic example of what it can do and only what it can do:

```py
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
```

Also, feel free to check [the full classification 
  example](https://github.com/haifeng-jin/readable-ml-framework/blob/main/examples/classification_example_with_readable_ml_framework.ipynb).

## Disclaimer

This repo is mainly for educational purposes only and no where near a
feature-complete ML framework. It is for people, who wants to learn the
internal mechanisms of ML frameworks, like TensorFlow, PyTorch, and JAX.

It implements the eager mode of execution with the tensor data structure and
operators in C++ and exposed with Python APIs. The operators are implemented
with multi-threading for speed optimization.

The code is structured in a way that is easiest for people to read. All complex
features, like sanity checks for function arguments, GPU support, distributed
training, data types of different precisions, asynchronous dispatch, compilers,
are not implemented.

## How to use

You can read the codebase in the following steps:

* Read the Python code in
  [`framework/tensor.py`](https://github.com/haifeng-jin/readable-ml-framework/blob/main/framework/tensor.py)
  to understand how `Tensor` works.
* Read the
  [`setup.py`](https://github.com/haifeng-jin/readable-ml-framework/blob/main/setup.py),
  [`framework/core/python_bind.py`](https://github.com/haifeng-jin/readable-ml-framework/blob/main/framework/core/python_bind.cpp)
  to understand how the Python and C++ interfacing works.
* Read
  [`framework/core/tensor.h`](https://github.com/haifeng-jin/readable-ml-framework/blob/main/framework/core/tensor.h)
  and
  [`framework/core/tensor.cpp`](https://github.com/haifeng-jin/readable-ml-framework/blob/main/framework/core/tensor.cpp)
  to understand how the underlying C++ implementation of the `Tensor` works.
* Read
  [`framework/core/ops.h`](https://github.com/haifeng-jin/readable-ml-framework/blob/main/framework/core/ops.h)
  and
  [`framework/core/ops.cpp`](https://github.com/haifeng-jin/readable-ml-framework/blob/main/framework/core/ops.cpp)
  to understand how the tensor operations are implemented in C++. This is the
  hardest part of the codebase. Feel free to skip all the details.
* Read the Python files under [`framework/ops/forward.py`](https://github.com/haifeng-jin/readable-ml-framework/blob/main/framework/ops/forward.py) and
  [`framework/ops/backward.py`](https://github.com/haifeng-jin/readable-ml-framework/blob/main/framework/ops/backward.py) understand how the C++ ops are wrapped in Python.
* Read
  [`framework/ops/op_record.py`](https://github.com/haifeng-jin/readable-ml-framework/blob/main/framework/ops/op_record.py)
  and
  [`framework/autograd.py`](https://github.com/haifeng-jin/readable-ml-framework/blob/main/framework/autograd.py)
  to understand how do we trace the ops and do backward propagation to compute
  the gradients.

## Install for development

I used a conda environment for easier setup.

Install the dependencies:

```
conda install -c conda-forge cxx-compiler clang-format
pip install -r requirements.txt
```

Install the project for dev mode:
```
pip install -e .
```
