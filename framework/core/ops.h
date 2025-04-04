// ops.h
#ifndef OPS_H
#define OPS_H
#include "tensor.h"

namespace ops {
void matmul(const Tensor &a, const Tensor &b, Tensor &output);
void add(const Tensor& a, const Tensor& b, Tensor &output);
void multiply(const Tensor &a, const Tensor &b, Tensor &output);
void relu(const Tensor& tensor, Tensor &output);
void softmax(const Tensor& tensor, Tensor &output);
void log(const Tensor& tensor, Tensor &output);

void sum(const Tensor& tensor, Tensor &output);
void sum_backward(const Tensor &output_grad, const Tensor &tensor, Tensor &input_grad);

};

#endif // OPS_H
