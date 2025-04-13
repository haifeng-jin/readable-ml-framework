// ops.h
#ifndef OPS_H
#define OPS_H
#include "tensor.h"

namespace ops {
/*
 * Forward operations:
 */

/**
 * @brief Performs matrix multiplication of two tensors.
 *
 * @param x: Tensor of shape (m, k). The first input tensor.
 * @param y: Tensor of shape (k, n). The second input tensor.
 * @param output: Tensor of shape (m, n). The output tensor.
 */
void matmul(const Tensor &x, const Tensor &y, Tensor &output);

/**
 * @brief Performs element-wise addition of a row vector to each row of a
 * matrix (broadcasting).
 * 
 * Example: if x is {{1.0, 2.0}, {3.0, 4.0}} and y is {5.0, 6.0}, output would
 * be {{1.0 + 5.0, 2.0 + 6.0}, {3.0 + 5.0, 4.0 + 6.0}}.
 *
 * @param x: Tensor of shape (m, n). The input matrix.
 * @param y: Tensor of shape (1, n). The row vector.
 * @param output: Tensor of shape (m, n). The output tensor.
 */
void add_row_broadcast(const Tensor& x, const Tensor& y, Tensor &output);

/**
 * @brief Performs element-wise in-place addition of two tensors.
 *
 * @param x: Tensor of the shape (m, n). The first input tensor (modified
 * in-place).
 * @param y: Tensor of the shape (m, n). The second input tensor.
 */
void add_element_wise_(Tensor &x, const Tensor &y);

/**
 * @brief Performs element-wise multiplication of two tensors.
 *
 * @param x: Tensor of shape (m, n). The first input tensor.
 * @param y: Tensor of shape (m, n). The second input tensor.
 * @param output: Tensor of shape (m, n). The output tensor.
 */
void multiply(const Tensor &x, const Tensor &y, Tensor &output);

/**
 * @brief Applies the Rectified Linear Unit (ReLU) activation function
 * element-wise.
 *
 * @param x: Tensor of shape (m, n). The first input tensor.
 * @param output: Tensor of shape (m, n). The output tensor.
 */
void relu(const Tensor& x, Tensor &output);

/**
 * @brief Applies the Softmax activation function along the last dimension.
 *
 * @param x: Tensor of shape (m, n). The first input tensor.
 * @param output: Tensor of shape (m, n). The output tensor.
 */
void softmax(const Tensor& x, Tensor &output);

/**
 * @brief Computes the natural logarithm (base e) of each element in the
 * tensor.
 *
 * @param x: Tensor of shape (m, n). The first input tensor.
 * @param output: Tensor of shape (m, n). The output tensor.
 */
void log(const Tensor& x, Tensor &output);

/**
 * @brief Computes the sum of all elements in the tensor, resulting in a scalar
 * tensor.
 *
 * @param x: Tensor of shape (m, n). The first input tensor.
 * @param output: Tensor of shape (1,). The output tensor.
 */
void sum(const Tensor &x, Tensor &output);

/*
 * Backward operations:
 */

/**
 * @brief Computes the gradients for the matrix multiplication operation.
 *
 * @param output_grad: Tensor of shape (m, n). Gradient of the loss with
 * respect to the output.
 * @param x: Tensor of shape (m, k). The first input tensor from the forward pass.
 * @param y: Tensor of shape (k, n). The second input tensor from the forward pass.
 * @param x_grad: Tensor of shape (m, k). Gradient of the loss with respect to x.
 * @param y_grad: Tensor of shape (k, n). Gradient of the loss with respect to y.
 */
void matmul_backward(const Tensor &output_grad, const Tensor &x, const Tensor &y,
                     Tensor &x_grad, Tensor &y_grad);

/**
 * @brief Computes the gradients for the row broadcasted addition operation.
 *
 * @param output_grad: Tensor of shape (m, n). Gradient of the loss with
 * respect to the output.
 * @param x: Tensor of shape (m, n). The first input tensor from the forward pass.
 * @param y: Tensor of shape (1, n). The second input tensor from the forward pass.
 * @param x_grad: Tensor of shape (m, n). Gradient of the loss with respect to x.
 * @param y_grad: Tensor of shape (1, n). Gradient of the loss with respect to y (summed over rows).
 */
void add_row_broadcast_backward(const Tensor &output_grad, const Tensor &x, const Tensor &y,
                                  Tensor &x_grad, Tensor &y_grad);

/**
 * @brief Computes the gradients for the element-wise multiplication operation.
 *
 * @param output_grad: Tensor of shape (m, n). Gradient of the loss with
 * respect to the output.
 * @param x: Tensor of shape (m, n). The first input tensor from the forward pass.
 * @param y: Tensor of shape (m, n). The second input tensor from the forward pass.
 * @param x_grad: Tensor of shape (m, n). Gradient of the loss with respect to x.
 * @param y_grad: Tensor of shape (m, n). Gradient of the loss with respect to y.
 */
void multiply_backward(const Tensor &output_grad, const Tensor &x, const Tensor &y,
                       Tensor &x_grad, Tensor &y_grad);

/**
 * @brief Computes the gradient for the ReLU activation function.
 *
 * @param output_grad: Tensor of shape (m, n). Gradient of the loss with
 * respect to the output.
 * @param x: Tensor of shape (m, n). The first input tensor from the forward pass.
 * @param x_grad: Tensor of shape (m, n). Gradient of the loss with respect to x.
 */
void relu_backward(const Tensor &output_grad, const Tensor &x,
                    Tensor &x_grad);

/**
 * @brief Computes the gradient for the Softmax activation function.
 *
 * @param output_grad: Tensor of shape (m, n). Gradient of the loss with
 * respect to the output.
 * @param x: Tensor of shape (m, n). The first input tensor from the forward pass.
 * @param x_grad: Tensor of shape (m, n). Gradient of the loss with respect to x.
 */
void softmax_backward(const Tensor &output_grad, const Tensor &x,
                       Tensor &x_grad);

/**
 * @brief Computes the gradient for the natural logarithm operation.
 *
 * @param output_grad: Tensor of shape (m, n). Gradient of the loss with
 * respect to the output.
 * @param x: Tensor of shape (m, n). The first input tensor from the forward pass.
 * @param x_grad: Tensor of shape (m, n). Gradient of the loss with respect to x.
 */
void log_backward(const Tensor &output_grad, const Tensor &x,
                    Tensor &x_grad);

/**
 * @brief Computes the gradient for the sum operation.
 *
 * @param output_grad: Tensor of shape (1,). Gradient of the loss with respect
 * to the output (the sum).
 * @param x: Tensor of shape (m, n). The first input tensor from the forward pass.
 * @param x_grad: Tensor of shape (m, n). Gradient of the loss with respect to x.
 */
void sum_backward(const Tensor &output_grad, const Tensor &x,
                    Tensor &x_grad);

};

#endif // OPS_H
