// ops.h
#ifndef OPS_H
#define OPS_H
#include "tensor.h"

namespace ops {
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor add(const Tensor& a, const Tensor& b);
Tensor relu(const Tensor& tensor);
Tensor softmax(const Tensor& tensor);
Tensor log(const Tensor& tensor);
Tensor sum(const Tensor& tensor);
};

#endif // OPS_H
