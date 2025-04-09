// ops.h
#ifndef OPS_H
#define OPS_H
#include "tensor.h"

namespace ops {
// Forward operations
void matmul(const Tensor &a, const Tensor &b, Tensor &output);
void add_row_broadcast(const Tensor& a, const Tensor& b, Tensor &output);
void add_element_wise_(Tensor &a, const Tensor &b);
void multiply(const Tensor &a, const Tensor &b, Tensor &output);
void relu(const Tensor& tensor, Tensor &output);
void softmax(const Tensor& tensor, Tensor &output);
void log(const Tensor& tensor, Tensor &output);
void sum(const Tensor& tensor, Tensor &output);

// Backward operations
void matmul_backward(const Tensor &output_grad, const Tensor &a, const Tensor &b, 
                     Tensor &a_grad, Tensor &b_grad);
void add_row_broadcast_backward(const Tensor &output_grad, const Tensor &a, const Tensor &b,
                  Tensor &a_grad, Tensor &b_grad);
void multiply_backward(const Tensor &output_grad, const Tensor &a, const Tensor &b,
                       Tensor &a_grad, Tensor &b_grad);
void relu_backward(const Tensor &output_grad, const Tensor &input,
                   Tensor &input_grad);
void softmax_backward(const Tensor &output_grad, const Tensor &input_data,
                      Tensor &input_grad);
void log_backward(const Tensor &output_grad, const Tensor &input,
                  Tensor &input_grad);
void sum_backward(const Tensor &output_grad, const Tensor &tensor, 
                  Tensor &input_grad);

};

#endif // OPS_H
