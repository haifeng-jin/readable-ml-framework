// ops.h
#ifndef OPS_H
#define OPS_H
#include "tensor.h"

Tensor matmul(const Tensor& a, const Tensor& b);
Tensor add_broadcast_row(const Tensor& a, const Tensor& b);
Tensor relu(const Tensor& tensor);

#endif // OPS_H
