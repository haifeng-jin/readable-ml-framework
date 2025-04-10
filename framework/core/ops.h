// ops.h
#ifndef OPS_H
#define OPS_H
#include "tensor.h"

namespace ops {
// Forward operations
void matmul(const Tensor &x, const Tensor &y, Tensor &output);
void add_row_broadcast(const Tensor& x, const Tensor& y, Tensor &output);
void add_element_wise_(Tensor &x, const Tensor &y);
void multiply(const Tensor &x, const Tensor &y, Tensor &output);
void relu(const Tensor& x, Tensor &output);
void softmax(const Tensor& x, Tensor &output);
void log(const Tensor& x, Tensor &output);
void sum(const Tensor &x, Tensor &output);

// Backward operations
void matmul_backward(const Tensor &output_grad, const Tensor &x, const Tensor &y,
                      Tensor &x_grad, Tensor &y_grad);
void add_row_broadcast_backward(const Tensor &output_grad, const Tensor &x, const Tensor &y,
                         Tensor &x_grad, Tensor &y_grad);
void multiply_backward(const Tensor &output_grad, const Tensor &x, const Tensor &y,
                          Tensor &x_grad, Tensor &y_grad);
void relu_backward(const Tensor &output_grad, const Tensor &x,
                         Tensor &x_grad);
void softmax_backward(const Tensor &output_grad, const Tensor &x,
                          Tensor &x_grad);
void log_backward(const Tensor &output_grad, const Tensor &x,
                        Tensor &x_grad);
void sum_backward(const Tensor &output_grad, const Tensor &x,
                        Tensor &x_grad);

};

#endif // OPS_H

