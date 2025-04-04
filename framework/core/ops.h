// ops.h
#ifndef OPS_H
#define OPS_H
#include "tensor.h"

namespace ops {
void matmul(const Tensor &a, const Tensor &b, Tensor &result);
void add(const Tensor& a, const Tensor& b, Tensor &result);
void relu(const Tensor& tensor, Tensor &result);
void softmax(const Tensor& tensor, Tensor &result);
void log(const Tensor& tensor, Tensor &result);
void sum(const Tensor& tensor, Tensor &result);
void multiply(const Tensor &a, const Tensor &b, Tensor &result);
};

#endif // OPS_H
