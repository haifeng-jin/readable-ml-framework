// ops.h
#ifndef OPS_H
#define OPS_H
#include "tensor.h"

namespace ops {
void matmul(const Tensor &a, const Tensor &b, Tensor &result);
Tensor add(const Tensor& a, const Tensor& b);
Tensor relu(const Tensor& tensor);
Tensor softmax(const Tensor& tensor);
Tensor log(const Tensor& tensor);
Tensor sum(const Tensor& tensor);
Tensor multiply(const Tensor &a, const Tensor &b);
};

#endif // OPS_H
