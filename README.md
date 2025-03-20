# Readable ML Framework

A machine learning framework with readable source code.

Machine learning frameworks are intimidating. The codebases are huge and
complex. It is almost impossible to pick up the source code and read to figure
out what is going on inside, when you write a machine learning model using it.

Fortunately, you have the "Readable ML Framework", which only contains a
limited number of files and all of which are well documented. It is just good
enough to implement a simple neural network to solve the MNIST dataset without
any extra features. You can easily understand all the basics of a ML framework
by reading it.

## Disclaimer

This repo is mainly for educational purposes only and no where near a
feature-complete ML framework. It is for people, who wants to learn the
internal mechanisms of ML frameworks, like TensorFlow, PyTorch, and JAX. It
implements the eager mode of execution with the tensor data structure and
operators in C++ and exposed with Python APIs. The operators are implemented
with multi-threading for speed optimization.

There are many features that we intentionally omitted to keep the codebase
simple enough to read, which includes GPU support, distributed computing,
compilers, SIMD support, etc.

## Install

```
conda install -c conda-forge cxx-compiler
```

```
pip install -r requirements.txt
pip install -e .
```
## Milestones

This is also a learning process for myself. So, here are the milestones:

- [ ] A basic `Tensor` class implemented in C++ and can accept NumPy array from
  Python and print back the content.
- [ ] matmul
- [ ] matmul backwards
- [ ] autodiff.
- [ ] more ops.
- [ ] Solve the MNIST dataset.
