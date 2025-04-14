#include "ops.h"
#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <numeric>
#include <thread>
#include <vector>

namespace py = pybind11;

// Nest all the operation functions under the ops to avoid naming conflicts.
namespace ops {

/**
 * @brief This function run tasks in parallel with multi-threading.
 *
 * @param total_work: The number of tasks that could run in parallel.
 * @param func: The function that runs an interval of tasks. The function
 * should accept two size_t type parameters, start and end, before the rest of
 * the arguments. Start and end marks the interval of tasks that the task
 * function should run.
 * @param args: The additional arguments to pass to the task function.
 */
template <typename Func, typename... Args>
void parallel_for(size_t total_work, Func &&func, Args &&...args) {
    // Determine the number of threads to create. To avoid being slow down by
    // the overhead of creating threads, we do not want to create too many
    // threads. We only need one thread for every 1000 tasks. The number of
    // threads should also not be greater than the maximum number of threads
    // supported by the CPU.
    size_t threads_needed = (total_work + 999) / 1000;
    size_t threads_supported =
        static_cast<size_t>(std::thread::hardware_concurrency());
    size_t num_threads = std::min(threads_supported, threads_needed);

    // A vector to store the threads.
    std::vector<std::future<void>> futures;

    // The number of tasks for each thread.
    size_t work_per_thread = total_work / num_threads;
    // Start of the interval.
    size_t start = 0;
    // End of the interval.
    size_t end = std::min(start + work_per_thread, total_work);

    // Launch all the threads asynchronously.
    while (start < total_work) {
        // Launch the thread for the current interval.
        futures.push_back(std::async(std::launch::async, func, start, end,
                                     std::forward<Args>(args)...));
        // Roll start and end forward to the next interval.
        start = end;
        end = std::min(start + work_per_thread, total_work);
    }

    // Sync all the threads. This is an obvious sub-optimal solution. To reach
    // maximum performance, we should not sync the threads until we really need
    // to write back the value to an external storage.
    for (auto &future : futures) {
        future.get();
    }
}

/*
 * Forward functions:
 */

void matmul_task(size_t start_row, size_t end_row, size_t num_x_cols,
                 size_t num_y_cols, const std::vector<float> &x_data,
                 const std::vector<float> &y_data,
                 std::vector<float> &output_data) {
    // Iterate through a specified range of rows of the first matrix (x).
    for (size_t i = start_row; i < end_row; ++i) {
        // For each row of x in the assigned range, iterate through all columns
        // of the second matrix (y).
        for (size_t j = 0; j < num_y_cols; ++j) {
            // This inner loop calculates the dot product (inner product) of
            // the i-th row of x and the j-th column of y. This is the
            // fundamental operation of matrix multiplication.
            float inner_product = 0.0f;
            for (size_t l = 0; l < num_x_cols; ++l) {
                inner_product +=
                    x_data[i * num_x_cols + l] * y_data[l * num_y_cols + j];
            }

            // The computed inner product is the element at the i-th row and
            // j-th column of the output matrix.
            output_data[i * num_y_cols + j] = inner_product;
        }
    }
}

/**
 * Performs parallel matrix multiplication of two tensors (matrices) x and y,
 * storing the result in output.
 *
 * output[i,j] is the inner-product of the ith row of x and the j th column of
 * y.
 *
 * This implementation employs a row-wise parallelization strategy. The rows of
 * the first matrix (x) are divided into chunks (implicitly by the parallel_for
 * construct), and each chunk is processed concurrently by invoking the
 * `matmul_task` function.
 *
 * Specifically, for a given sub-range of rows [start_row, end_row) of x,
 * `matmul_task` computes the corresponding sub-block of the output matrix by
 * multiplying these rows with the entirety of the second matrix (y). If a task
 * is assigned the row slice x[start_row:end_row,:], it calculates
 * output[start_row:end_row,:].
 *
 * The choice of row-wise sharding simplifies the parallelization logic.
 * However, it's worth noting that more advanced parallel matrix multiplication
 * techniques often involve sharding both input matrices (x and y) to achieve
 * finer-grained parallelism and potentially better data locality, especially
 * on distributed memory systems. For instance, a common approach involves
 * assigning tasks to compute sub-blocks of the output matrix resulting from
 * the multiplication of a row block of x and a column block of y (e.g., if a
 * thread processes x[a:b,:] and y[:,c:d], it would produce output[a:b,c:d]).
 * These sub-blocks are then combined (tiled) to form the final output matrix.
 */
void matmul(const Tensor &x, const Tensor &y, Tensor &output) {
    size_t num_x_rows = x.shape[0];
    size_t num_x_cols = x.shape[1];
    size_t num_y_cols = y.shape[1];

    parallel_for(
        // Number of tasks that are parallelizable.
        num_x_rows,
        // The function for each thread.
        matmul_task,
        // More args for the function.
        num_x_cols, num_y_cols, std::ref(x.data), std::ref(y.data),
        std::ref(output.data));
}

void add_row_broadcast_task(size_t start, size_t end, size_t num_y_cols,
                            const std::vector<float> &x_data,
                            const std::vector<float> &y_data,
                            std::vector<float> &output_data) {
    // Iterate the elements in the given interval.
    for (size_t i = start; i < end; ++i) {
        output_data[i] = x_data[i] + y_data[i % num_y_cols];
    }
}

/**
 * Add vector y to matrix x by broadcasting the vector to all the rows.
 *
 * So output[i, j] = x[i, j] + y[j].
 * If we flatten all of them, we get:
 *
 * for i in range(num_rows):
 *     for j in range(num_cols):
 *         output[i * num_cols + j] = x[i * num_cols + j] + y[j],
 *
 * which is equivalent to:
 *
 * for i in range(num_rows * num_cols):
 *     output[i] = x[i] + y[i % num_cols]
 *
 * So all the add operations are parallelizable.
 */
void add_row_broadcast(const Tensor &x, const Tensor &y, Tensor &output) {
    size_t num_x_rows = x.shape[0];
    size_t num_y_cols = y.data.size();

    parallel_for(
        // Number of tasks that are parallelizable.
        x.data.size(),
        // The function for each thread.
        add_row_broadcast_task,
        // More args for the function.
        num_y_cols, std::ref(x.data), std::ref(y.data), std::ref(output.data));
}

void add_element_wise_task(size_t start, size_t end, std::vector<float> &x_data,
                           const std::vector<float> &y_data) {
    // Iterate the elements in the given interval.
    for (size_t i = start; i < end; ++i) {
        x_data[i] += y_data[i];
    }
}

/**
 * Perform element-wise in-place add on two tensors.
 *
 * Since it is an element-wise operation, the shapes are not used. The tensors
 * can be seen as flattened. So all the add operations are parallelizable.
 */
void add_element_wise_(Tensor &x, const Tensor &y) {
    parallel_for(
        // Number of tasks that are parallelizable.
        x.data.size(),
        // The function for each thread.
        add_element_wise_task,
        // More args for the function.
        std::ref(x.data), std::ref(y.data));
}

void multiply_task(size_t start, size_t end, const std::vector<float> &x_data,
                   const std::vector<float> &y_data,
                   std::vector<float> &output_data) {
    // Iterate the elements in the given interval.
    for (size_t i = start; i < end; ++i) {
        output_data[i] = x_data[i] * y_data[i];
    }
}

/**
 * Perform element-wise multiply on two tensors.
 *
 * Since it is an element-wise operation, the shapes are not used. The tensors
 * can be seen as flattened. So all the multiply operations are parallelizable.
 */
void multiply(const Tensor &x, const Tensor &y, Tensor &output) {
    parallel_for(
        // Number of tasks that are parallelizable.
        output.data.size(),
        // The function for each thread.
        multiply_task,
        // More args for the function.
        std::ref(x.data), std::ref(y.data), std::ref(output.data));
}

void relu_task(size_t start, size_t end, const std::vector<float> &x_data,
               std::vector<float> &output_data) {
    // Iterate the elements in the given interval.
    for (size_t i = start; i < end; ++i) {
        output_data[i] = std::max(0.0f, x_data[i]);
    }
}

/**
 * Perform element-wise relu on the tensor.
 *
 * Since it is an element-wise operation, the shapes are not used. The tensor
 * can be seen as flattened. So all the relu operations are parallelizable.
 */
void relu(const Tensor &x, Tensor &output) {
    parallel_for(
        // Number of tasks that are parallelizable.
        output.data.size(),
        // The function for each thread.
        relu_task,
        // More args for the function.
        std::ref(x.data), std::ref(output.data));
}

void softmax_task(size_t start_row, size_t end_row, size_t num_x_cols,
                  const std::vector<float> &x_data,
                  std::vector<float> &output_data) {
    for (size_t i = start_row; i < end_row; ++i) {
        size_t row_start = i * num_x_cols;
        size_t row_end = row_start + num_x_cols;
        float max_val = -std::numeric_limits<float>::infinity();

        // Find the maximum value in the row for numerical stability
        for (size_t j = row_start; j < row_end; ++j) {
            if (x_data[j] > max_val) {
                max_val = x_data[j];
            }
        }

        float sum_exp = 0.0f;
        // Calculate the exponential of each element and the sum of exponentials
        for (size_t j = row_start; j < row_end; ++j) {
            output_data[j] =
                std::exp(x_data[j] - max_val); // Subtract max_val for stability
            sum_exp += output_data[j];
        }

        // Normalize the row by dividing by the sum of exponentials
        for (size_t j = row_start; j < row_end; ++j) {
            output_data[j] /= sum_exp;
        }
    }
}

/**
 * Applies the softmax function to each row of the input tensor independently.
 *
 * Softmax converts a vector of raw scores into a probability distribution. For
 * each row in the input tensor, the softmax function is applied such that the
 * resulting elements are in the range [0, 1] and sum to 1. This is commonly
 * used in the output layer of classification neural networks.
 *
 * Mathematically, for an input vector x = [x_1, x_2, ..., x_C]
 * the softmax function produces
 * an output vector S(x) = [S(x_1), S(x_2), ..., S(x_C)] where each
 * element is calculated as: S(x_i) = exp(x_i) / sum_{for j=1 to C} exp(x_j)
 * where C is the number of columns.
 *
 * This implementation leverages row-wise parallelization. The rows of the
 * input tensor `x` are divided into chunks (implicitly by the `parallel_for`
 * construct), and each chunk is processed concurrently by the `softmax_task`
 * function. Each invocation of `softmax_task` computes the softmax for a
 * subset of rows.
 *
 * Numerical Stability:
 * To prevent potential overflow issues when calculating exponentials of large
 * numbers, the implementation incorporates a standard trick: it subtracts the
 * maximum value of each row from all elements in that row before
 * exponentiation. This does not change the result of the softmax function due
 * to the normalization step but significantly improves numerical stability.
 */
void softmax(const Tensor &x, Tensor &output) {
    size_t num_x_rows = x.shape[0];
    size_t num_x_cols = x.shape[1];

    parallel_for(
        // Number of tasks that are parallelizable.
        num_x_rows,
        // The function for each thread.
        softmax_task,
        // More args for the function.
        num_x_cols, std::ref(x.data), std::ref(output.data));
}

void log_task(size_t start, size_t end, const std::vector<float> &x_data,
              std::vector<float> &output_data) {
    // Iterate the elements in the given interval.
    for (size_t i = start; i < end; ++i) {
        // If the value is too small (less than 1e-8), we will use log(1e-8)
        // to avoid log(0), which is -inf.
        output_data[i] =
            (x_data[i] > 1e-8) ? std::log(x_data[i]) : std::log(1e-8);
    }
}

/**
 * Perform element-wise log on the tensor.
 *
 * Since it is an element-wise operation, the shapes are not used. The tensor
 * can be seen as flattened. So all the log operations are parallelizable.
 */
void log(const Tensor &x, Tensor &output) {
    parallel_for(
        // Number of tasks that are parallelizable.
        output.data.size(),
        // The function for each thread.
        log_task,
        // More args for the function.
        std::ref(x.data), std::ref(output.data));
}

void sum_task(size_t start_row, size_t end_row, size_t num_x_cols,
              const std::vector<float> &x_data,
              std::vector<float> &output_data) {
    for (size_t i = start_row; i < end_row; ++i) {
        output_data[i] = 0.0f;
        for (size_t j = 0; j < num_x_cols; ++j) {
            output_data[i] += x_data[i * num_x_cols + j];
        }
    }
}

/**
 * Computes the sum of all elements within a given tensor.
 *
 * This implementation utilizes a row-wise parallel reduction strategy. The
 * input tensor (x) is conceptually divided into chunks of rows (implicitly
 * managed by the `parallel_for` construct). Each chunk is processed
 * concurrently by the `sum_task` function, which calculates the sum of
 * elements within those rows.  The results of these parallel tasks are stored
 * in an intermediate vector (`partial_sum`), where each element represents the
 * sum of a corresponding row from the input tensor.
 *
 * Finally, the elements of this `partial_sum` vector are aggregated (summed
 * together) to produce the final total sum of all elements in the original
 * tensor. This final sum is stored in the first element of the output tensor's
 * data.
 *
 * While this approach is straightforward to implement and demonstrates basic
 * parallelization, it involves an intermediate reduction step (summing the row
 * sums). More advanced parallel reduction techniques can directly compute the
 * global sum with potentially higher efficiency and finer-grained parallelism.
 *
 * For instance, a different strategy could involve flattening the tensor and
 * dividing the resulting linear sequence into independent intervals. Each
 * interval's sum could be computed in parallel, and these partial sums could
 * then be combined using a parallel reduction tree to obtain the final result
 * more efficiently. This alternative avoids the explicit intermediate storage
 * of row sums and can offer better workload balancing and data locality.
 */
void sum(const Tensor &x, Tensor &output) {
    size_t num_x_rows = x.shape[0];
    size_t num_x_cols = x.shape[1];

    // Storing the sum of each row.
    std::vector<float> partial_sum(num_x_rows);
    parallel_for(
        // Number of tasks that are parallelizable.
        num_x_rows,
        // The function for each thread.
        sum_task,
        // More args for the function.
        num_x_cols, std::ref(x.data), std::ref(partial_sum));
    output.data[0] =
        std::accumulate(partial_sum.begin(), partial_sum.end(), 0.0f);
}

/*
 * Backward functions:
 */

void matmul_backward_task_x(size_t start_row, size_t end_row, size_t num_x_rows,
                            size_t num_x_cols, size_t num_y_cols,
                            const std::vector<float> &output_grad_data,
                            const std::vector<float> &y_data,
                            std::vector<float> &x_grad_data) {
    // d(loss)/d(x) = d(loss)/d(output) * transpose(y)
    // Iterate the rows of x in the interval.
    for (size_t i = start_row; i < end_row; ++i) {
        // Iterate the columns of x.
        for (size_t j = 0; j < num_x_cols; ++j) {
            // x_grad[i, j] is the inner product of ith row of output_grad and
            // jth row of y (jth column of y transpose).
            float inner_product = 0.0f;
            for (size_t l = 0; l < num_y_cols; ++l) {
                inner_product += output_grad_data[i * num_y_cols + l] *
                                 y_data[j * num_y_cols + l];
            }
            x_grad_data[i * num_x_cols + j] = inner_product;
        }
    }
}

void matmul_backward_task_y(size_t start_col, size_t end_col, size_t num_x_rows,
                            size_t num_x_cols, size_t num_y_cols,
                            const std::vector<float> &output_grad_data,
                            const std::vector<float> &x_data,
                            std::vector<float> &y_grad_data) {
    // d(loss)/d(y) = transpose(x) * d(loss)/d(output)
    // Iterate the columns of y in the interval.
    for (size_t j = start_col; j < end_col; ++j) {
        // Iterate the columns of x.
        for (size_t i = 0; i < num_x_cols; ++i) {
            // y_grad[i, j] is the inner-product of ith column of X (or the ith
            // row of X transpose) and the jth column of output_grad.
            float inner_product = 0.0f;
            for (size_t l = 0; l < num_x_rows; ++l) {
                inner_product += x_data[l * num_x_cols + i] *
                                 output_grad_data[l * num_y_cols + j];
            }
            y_grad_data[i * num_y_cols + j] = inner_product;
        }
    }
}

/**
 * The backward function of matrix multiplication.
 *
 * Given the gradients of the output of the multiplication, we want to get the
 * gradients of the inputs of the multiplication. Or more formally, given
 * d(loss)/d(output), where output = matmul(x, y), we want to compute
 * d(loss)/d(x), and d(loss)/d(y).
 *
 * Because loss is a scalar, d(loss)/d(output), d(loss)/d(x), and d(loss)/d(y),
 * are of the same shape of output, x, and y.
 *
 * According to matrix calculus rules:
 * d(loss)/d(x) = d(loss)/d(output) * transpose(y)
 * d(loss)/d(y) = transpose(x) * d(loss)/d(output)
 *
 * So, at its core, the backward of matmul is basically, two matmul operations.
 * We implemented it the same way as we did for the matmul operation.
 */
void matmul_backward(const Tensor &output_grad, const Tensor &x,
                     const Tensor &y, Tensor &x_grad, Tensor &y_grad) {
    size_t num_x_rows = x.shape[0];
    size_t num_x_cols = x.shape[1];
    size_t num_y_cols = y.shape[1];

    parallel_for(
        // Number of tasks that are parallelizable.
        num_x_rows,
        // The function for each thread.
        matmul_backward_task_x,
        // More args for the function.
        num_x_rows, num_x_cols, num_y_cols, std::ref(output_grad.data),
        std::ref(y.data), std::ref(x_grad.data));

    parallel_for(
        // Number of tasks that are parallelizable.
        num_y_cols,
        // The function for each thread.
        matmul_backward_task_y,
        // More args for the function.
        num_x_rows, num_x_cols, num_y_cols, std::ref(output_grad.data),
        std::ref(x.data), std::ref(y_grad.data));
}

void add_row_broadcast_backward_task_x(
    size_t start, size_t end, const std::vector<float> &output_grad_data,
    std::vector<float> &x_grad_data) {
    for (size_t i = start; i < end; ++i) {
        // Simply copy output_grad to x_grad.
        x_grad_data[i] = output_grad_data[i];
    }
}

void add_row_broadcast_backward_task_y(
    size_t start, size_t end, size_t num_y_cols,
    const std::vector<float> &output_grad_data,
    std::vector<float> &y_grad_data) {
    for (size_t i = start; i < end; ++i) {
        // Directly add the output_grad value to the corresponding column of
        // y_grad. y_grad was initialized to be all zero.
        y_grad_data[i % num_y_cols] += output_grad_data[i];
    }
}

/**
 * The backward function of adding a vector to each row of the matrix.
 *
 * d(output[i,j])/d(x[i,j]) is 1 because output[i,j] = x[i,j] + y[1,j], and if
 * f(x) = x + C, then f'(x) = 1.
 *
 * x_grad[i,j]) = d(loss)/d(x[i,j])
 *              = ( d(loss)/d(output[i,j]) ) * ( d(output[i,j])/d(x[i,j]) )
 *              = ( output_grad[i,j]) ) * ( 1 ) = output_grad[i,j]
 * So, x_grad is just a simple copy of output_grad.
 *
 * However, for grad_y, since y is added to each row of x to produce output,
 * there are multiple gradients to y. When y is added to the ith row of x, the
 * corresponding gradients is output_grad[i]. We need to sum up the gradients
 * of all the rows to produce y_grad.
 *
 * So, y_grad is to sum output_grad across the rows and along the columns.
 * Similar to the implementation of the forward function, we can just use a
 * flattened view of y_grad and output_grad and use % to get the index of the
 * element.
 */
void add_row_broadcast_backward(const Tensor &output_grad, const Tensor &x,
                                const Tensor &y, Tensor &x_grad,
                                Tensor &y_grad) {
    // Copy output_grad to x_grad.
    parallel_for(
        // Number of tasks that are parallelizable.
        x.data.size(),
        // The function for each thread.
        add_row_broadcast_backward_task_x,
        // More args for the function.
        std::ref(output_grad.data), std::ref(x_grad.data));

    // For y_grad, sum the gradients across the rows and along each column.
    parallel_for(
        // Number of tasks that are parallelizable.
        output_grad.data.size(),
        // The function for each thread.
        add_row_broadcast_backward_task_y,
        // More args for the function.
        y.data.size(), std::ref(output_grad.data), std::ref(y_grad.data));
}

void multiply_backward_task_x(size_t start, size_t end,
                              const std::vector<float> &output_grad_data,
                              const std::vector<float> &y_data,
                              std::vector<float> &x_grad_data) {
    for (size_t i = start; i < end; ++i) {
        x_grad_data[i] = output_grad_data[i] * y_data[i];
    }
}

void multiply_backward_task_y(size_t start, size_t end,
                              const std::vector<float> &output_grad_data,
                              const std::vector<float> &x_data,
                              std::vector<float> &y_grad_data) {
    for (size_t i = start; i < end; ++i) {
        y_grad_data[i] = output_grad_data[i] * x_data[i];
    }
}

/**
 * The backward function of element-size multiply.
 *
 * If f(x) = x * C, then f'(x) = C.
 * So, d(output[i,j])/d(x[i,j]) = y[i,j], where output[i,j] = x[i,j] * y[i,j].
 * x_grad[i,j] = d(loss)/d(x[i,j])
 *             = ( d(loss)/d(output[i,j]) ) * ( d(output[i,j])/d(x[i,j]) )
 *             = output_grad[i,j] * y[i,j]
 *
 * Similarly, y_grad[i,j] = output[i,j] * x[i,j]
 *
 * So, it is still element-wise multiplication in the backward function. They
 * can be done in a flattened view of the tensors similar to the forward
 * function.
 */
void multiply_backward(const Tensor &output_grad, const Tensor &x,
                       const Tensor &y, Tensor &x_grad, Tensor &y_grad) {
    parallel_for(
        // Number of tasks that are parallelizable.
        x.data.size(),
        // More args for the function.
        multiply_backward_task_x,
        // The function for each thread.
        std::ref(output_grad.data), std::ref(y.data), std::ref(x_grad.data));

    parallel_for(
        // Number of tasks that are parallelizable.
        y.data.size(),
        // The function for each thread.
        multiply_backward_task_y,
        // More args for the function.
        std::ref(output_grad.data), std::ref(x.data), std::ref(y_grad.data));
}

void relu_backward_task(size_t start, size_t end,
                        const std::vector<float> &output_grad_data,
                        const std::vector<float> &x_data,
                        std::vector<float> &x_grad_data) {
    for (size_t i = start; i < end; ++i) {
        // x_grad[i,j] = output_grad[i,j] * (0 or 1)
        // We are using a flattened view of the tensor, so there is no column
        // index j.
        x_grad_data[i] = x_data[i] > 0 ? output_grad_data[i] : 0.0f;
    }
}

/**
 * The backward function of relu.
 *
 * relu can be seen as a element-wise multiplication between x and another
 * matrix y full of 0s and 1s. If x[i,j] is negative, y[i,j] is 0, otherwise, 1.
 *
 * We will do exactly the same as the backward function of element-wise
 * multiplication, except that there is no y_grad needed.
 */
void relu_backward(const Tensor &output_grad, const Tensor &x, Tensor &x_grad) {
    parallel_for(
        // Number of tasks that are parallelizable.
        x.data.size(),
        // The function for each thread.
        relu_backward_task,
        // More args for the function.
        std::ref(output_grad.data), std::ref(x.data), std::ref(x_grad.data));
}

/**
 * Computes the backward pass of the softmax function for a subset of rows.
 *
 * Quick recap:
 * If the input is an vector x, and s = softmax(x), then
 * s[i] = exp(x[i]) / sum(exp(x[j]) for j in range(len(x)))
 * For numerical stability, we usually normalize x by subtracing max(x) on
 * every element.
 *
 * According to matrix calculus:
 * x_grad = d(loss)/d(x) = matmul(Jacobian(output) * (d(loss)/d(output))),
 * where Jacobian is the Jacobian-vector product (JVP).
 *
 * How to compute JVP, namely matrix J:
 *
 * J[i][j] = s[i]*(1-s[i]) if i == j (the element is on the diagonal)
 * J[i][j] = -s[i]s[j] if i != j (the element is not on the diagonal)
 *
 * We can simplify it to:
 *
 * J[i][j] = -s[i]s[j] for all the elements.
 * J[i][j] += s[i] for all the elements on the diagonal.
 *
 * Let's simplify the case by saying x_grad_row is one row of the x_grad
 * matrix, and output_grad_row is one row of the output_grad matrix.
 * So, x_grad_row = matmul(JVP, output_grad_row).
 * x_grad_row[j] = inner product of jth row of JVP and the output_grad_row.
 * So, we are using one row of JVP at a time.
 *
 * With this, we can compute the JVP row by row instead of all at once to save
 * memory.  With each row, we compute the inner product and save it in the
 * x_grad.
 */
void softmax_backward_task(size_t start_row, size_t end_row, size_t n,
                           const std::vector<float> &output_grad_data,
                           const std::vector<float> &x_data,
                           std::vector<float> &x_grad_data) {
    for (size_t i = start_row; i < end_row; ++i) {
        size_t row_start = i * n;
        float max_val = -std::numeric_limits<float>::infinity();

        // We first compute the softmax forward pass on the fly.

        // Compute the max value in x.
        for (size_t j = 0; j < n; ++j) {
            if (x_data[row_start + j] > max_val) {
                max_val = x_data[row_start + j];
            }
        }

        float sum_exp = 0.0f;
        // Calculate the exponential of each element and the sum of exponentials
        for (size_t j = 0; j < n; ++j) {
            sum_exp += std::exp(x_data[row_start + j] -
                                max_val); // Subtract max_val for stability
        }

        // Now, we have max_val, and sum_exp, which will be used to compute any
        // s[i] on the fly.

        // Iterate the elements in one row of x_grad.
        for (size_t j = 0; j < n; ++j) {
            float inner_product = 0.0f;
            // Compute s[j]
            float softmax_j =
                std::exp(x_data[row_start + j] - max_val) / sum_exp;
            // Iterate the elements in one row of JVP and output_grad.
            for (size_t k = 0; k < n; ++k) {
                // Compute s[k]
                float softmax_k =
                    std::exp(x_data[row_start + k] - max_val) / sum_exp;
                // Compute J[j][k]
                float jacobian_jk =
                    softmax_j * ((j == k ? 1.0f : 0.0f) - softmax_k);
                // Add J[j][k] * output_grad_row[k] to the inner-product.
                inner_product += jacobian_jk * output_grad_data[row_start + k];
            }
            // Assign the inner-product to x_grad[i][j].
            x_grad_data[row_start + j] = inner_product;
        }
    }
}

/**
 * The backward function of softmax.
 *
 * Given the gradient of the loss w.r.t. the softmax output (output_grad),
 * and the input tensor to the softmax (x), computes the gradient of the loss
 * w.r.t. the input x (x_grad).
 *
 * Although it's more efficient to compute this using the already computed
 * softmax output, this implementation opts to recompute softmax on the fly.
 * This strategy trades compute for memory — a strategy often used in
 * large-scale models to reduce memory usage (e.g., when softmax output is not
 * stored or is expensive to fetch from memory like HBM on GPUs).
 */
void softmax_backward(const Tensor &output_grad, const Tensor &x,
                      Tensor &x_grad) {
    size_t num_x_rows = x.shape[0];
    size_t num_x_cols = x.shape[1];

    parallel_for(
        // Number of tasks that are parallelizable.
        num_x_rows,
        // The function for each thread.
        softmax_backward_task,
        // More args for the function.
        num_x_cols, std::ref(output_grad.data), std::ref(x.data),
        std::ref(x_grad.data));
}

void log_backward_task(size_t start, size_t end,
                       const std::vector<float> &output_grad_data,
                       const std::vector<float> &x_data,
                       std::vector<float> &x_grad_data) {
    for (size_t i = start; i < end; ++i) {
        // Ensure not divided by zero.
        float x = std::max(x_data[i], 1e-8f);
        x_grad_data[i] = output_grad_data[i] / x;
    }
}

/**
 * The backward function for element-wise log.
 *
 * If f(x) = log(x), then f'(x) = 1/x.
 * So, d(output[i,j])/d(x[i,j]) = 1/x[i,j], where output[i,j] = log(x[i,j]).
 * x_grad[i,j] = d(loss)/d(x[i,j])
 *             = ( d(loss)/d(output[i,j]) ) * ( d(output[i,j])/d(x[i,j]) )
 *             = output_grad[i,j] / x[i,j]
 *
 * Since it is an element-wise operation, we can use the flattened views of the
 * tensors. We don't need to tell the row and column indices.
 */
void log_backward(const Tensor &output_grad, const Tensor &x, Tensor &x_grad) {
    parallel_for(
        // Number of tasks that are parallelizable.
        x.data.size(),
        // The function for each thread.
        log_backward_task,
        // More args for the function.
        std::ref(output_grad.data), std::ref(x.data), std::ref(x_grad.data));
}

void sum_backward_task(size_t start, size_t end, float output_grad_scalar,
                       std::vector<float> &x_grad_data) {
    for (size_t i = start; i < end; i++) {
        x_grad_data[i] = output_grad_scalar;
    }
}

/**
 * The backward function of sum all elements.
 *
 * If f(x) = x + C, then f'(x) = 1.
 * So, d(sum)/d(x[i,j]) = 1, where sum = x[i,j] + (all other elements).
 * x_grad[i,j] = d(loss)/d(x[i,j])
 *             = ( d(loss)/d(sum) ) * ( d(sum)/d(x[i,j]) )
 *             = output_grad * 1 = output_grad
 * Note that the output_grad and sum are two scalars.
 *
 * So, we can just fill x_grad with the output_grad scalar.
 *
 * Since there is no need to know the row and column indices during filling the
 * value, we use the flattened view of the tensor.
 */
void sum_backward(const Tensor &output_grad, const Tensor &x, Tensor &x_grad) {
    parallel_for(
        // Number of tasks that are parallelizable.
        x.data.size(),
        // The function for each thread.
        sum_backward_task,
        // More args for the function.
        output_grad.data[0], std::ref(x_grad.data));
}

}; // namespace ops
