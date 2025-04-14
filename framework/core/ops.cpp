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
    // Iterate the given element range.
    for (size_t i = start; i < end; ++i) {
        x_data[i] += y_data[i];
    }
}

/**
 * Perform element-wise in-place add on two tensors.
 *
 * Since it is an element-wise operation, the shapes are not used. The tensors
 * can be seen as flattened. So all the add operations are
 * parallelizable.
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
    for (size_t i = start; i < end; ++i) {
        output_data[i] = x_data[i] * y_data[i];
    }
}

void multiply(const Tensor &x, const Tensor &y, Tensor &output) {
    parallel_for(
        // Number of tasks that are parallelizable.
        output.data.size(),
        // The function for each thread.
        multiply_task,
        // More args for the function.
        std::ref(x.data), std::ref(y.data), std::ref(output.data));
}

// Function to apply ReLU to a subset of the tensor
void relu_task(size_t start, size_t end, const std::vector<float> &x_data,
               std::vector<float> &output_data) {
    for (size_t i = start; i < end; ++i) {
        output_data[i] = std::max(0.0f, x_data[i]);
    }
}

void relu(const Tensor &x, Tensor &output) {
    parallel_for(
        // Number of tasks that are parallelizable.
        output.data.size(),
        // The function for each thread.
        relu_task,
        // More args for the function.
        std::ref(x.data), std::ref(output.data));
}

// Function to apply Softmax to a subset of rows
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
    for (size_t i = start; i < end; ++i) {
        output_data[i] =
            (x_data[i] > 1e-8) ? std::log(x_data[i]) : std::log(1e-8);
    }
}

void log(const Tensor &x, Tensor &output) {
    parallel_for(
        // Number of tasks that are parallelizable.
        output.data.size(),
        // The function for each thread.
        log_task,
        // More args for the function.
        std::ref(x.data), std::ref(output.data));
}

// Function to calculate the sum of a subset of the tensor elements
void sum_task(size_t start_row, size_t end_row, size_t n,
              const std::vector<float> &data, std::vector<float> &output_data) {
    for (size_t i = start_row; i < end_row; ++i) {
        output_data[i] = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            output_data[i] += data[i * n + j];
        }
    }
}

void sum(const Tensor &x, Tensor &output) {
    std::vector<float> partial_sum(x.shape[0]);
    parallel_for(x.shape[0], sum_task, x.shape[1], std::ref(x.data),
                 std::ref(partial_sum));
    output.data[0] =
        std::accumulate(partial_sum.begin(), partial_sum.end(), 0.0f);
}

/*
 * Backward functions:
 */

// Function to perform matrix multiplication backward pass for a subset of rows
void matmul_backward_task_a(size_t start_row, size_t end_row, size_t m,
                            size_t n, size_t k,
                            const std::vector<float> &output_grad_data,
                            const std::vector<float> &x_data,
                            std::vector<float> &x_grad_data) {
    // Computing gradient with respect to A: dL/dA = dL/dC * B^T
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < k; ++j) {
            float sum = 0.0f;
            for (size_t l = 0; l < n; ++l) {
                sum += output_grad_data[i * n + l] * x_data[j * n + l];
            }
            x_grad_data[i * k + j] = sum;
        }
    }
}

void matmul_backward_task_b(size_t start_col, size_t end_col, size_t m,
                            size_t n, size_t k,
                            const std::vector<float> &output_grad_data,
                            const std::vector<float> &x_data,
                            std::vector<float> &y_grad_data) {
    // Computing gradient with respect to B: dL/dB = A^T * dL/dC
    for (size_t j = start_col; j < end_col; ++j) {
        for (size_t i = 0; i < k; ++i) {
            float sum = 0.0f;
            for (size_t l = 0; l < m; ++l) {
                sum += x_data[l * k + i] * output_grad_data[l * n + j];
            }
            y_grad_data[i * n + j] = sum;
        }
    }
}

void matmul_backward(const Tensor &output_grad, const Tensor &x,
                     const Tensor &y, Tensor &x_grad, Tensor &y_grad) {
    size_t m = x.shape[0];
    size_t k = x.shape[1];
    size_t n = y.shape[1];

    // Calculate gradients for A and B in parallel
    parallel_for(m, matmul_backward_task_a, m, n, k, std::ref(output_grad.data),
                 std::ref(y.data), std::ref(x_grad.data));

    parallel_for(n, matmul_backward_task_b, m, n, k, std::ref(output_grad.data),
                 std::ref(x.data), std::ref(y_grad.data));
}

// Function to perform add backward pass for a subset of columns
void add_row_broadcast_backward_task_b(
    size_t start_col, size_t end_col, size_t m, size_t n,
    const std::vector<float> &output_grad_data,
    std::vector<float> &y_grad_data) {
    // For y_grad, we need to sum the gradients across all rows for each column
    for (size_t j = start_col; j < end_col; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < m; ++i) {
            sum += output_grad_data[i * n + j];
        }
        y_grad_data[j] = sum;
    }
}

void add_row_broadcast_backward(const Tensor &output_grad, const Tensor &x,
                                const Tensor &y, Tensor &x_grad,
                                Tensor &y_grad) {
    size_t m = x.shape[0];
    size_t n = x.shape[1];

    // For x_grad, simply copy the output gradient as it flows through unchanged
    std::copy(output_grad.data.begin(), output_grad.data.end(),
              x_grad.data.begin());

    // For y_grad, sum the gradients across each column
    parallel_for(n, add_row_broadcast_backward_task_b, m, n,
                 std::ref(output_grad.data), std::ref(y_grad.data));
}

// Function to perform element-wise multiply backward pass
void multiply_backward_task_a(size_t start, size_t end,
                              const std::vector<float> &output_grad_data,
                              const std::vector<float> &x_data,
                              std::vector<float> &x_grad_data) {
    for (size_t i = start; i < end; ++i) {
        x_grad_data[i] = output_grad_data[i] * x_data[i];
    }
}

void multiply_backward_task_b(size_t start, size_t end,
                              const std::vector<float> &output_grad_data,
                              const std::vector<float> &x_data,
                              std::vector<float> &y_grad_data) {
    for (size_t i = start; i < end; ++i) {
        y_grad_data[i] = output_grad_data[i] * x_data[i];
    }
}

void multiply_backward(const Tensor &output_grad, const Tensor &x,
                       const Tensor &y, Tensor &x_grad, Tensor &y_grad) {
    // For element-wise multiplication, gradients are calculated as:
    // dL/dA = dL/dOutput * B
    // dL/dB = dL/dOutput * A

    parallel_for(x.data.size(), multiply_backward_task_a,
                 std::ref(output_grad.data), std::ref(y.data),
                 std::ref(x_grad.data));

    parallel_for(y.data.size(), multiply_backward_task_b,
                 std::ref(output_grad.data), std::ref(x.data),
                 std::ref(y_grad.data));
}

// Function to apply ReLU backward pass for a subset of the tensor
void relu_backward_task(size_t start, size_t end,
                        const std::vector<float> &output_grad_data,
                        const std::vector<float> &x_data,
                        std::vector<float> &x_grad_data) {
    for (size_t i = start; i < end; ++i) {
        // Gradient is output_grad if input was positive, 0 otherwise
        x_grad_data[i] = x_data[i] > 0 ? output_grad_data[i] : 0.0f;
    }
}

void relu_backward(const Tensor &output_grad, const Tensor &x, Tensor &x_grad) {
    parallel_for(x.data.size(), relu_backward_task, std::ref(output_grad.data),
                 std::ref(x.data), std::ref(x_grad.data));
}

// Function to apply Softmax backward pass for a subset of rows
void softmax_backward_task(size_t start_row, size_t end_row, size_t n,
                           const std::vector<float> &output_grad_data,
                           const std::vector<float> &x_data,
                           std::vector<float> &x_grad_data) {
    for (size_t i = start_row; i < end_row; ++i) {
        size_t row_start = i * n;
        float max_val = -std::numeric_limits<float>::infinity();
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

        // For each row, calculate the Jacobian vector product
        for (size_t j = 0; j < n; ++j) {
            float grad_sum = 0.0f;
            float softmax_j =
                std::exp(x_data[row_start + j] - max_val) / sum_exp;
            for (size_t k = 0; k < n; ++k) {
                // Jacobian of softmax: J[j,k] = s[j]*(delta[j,k] - s[k])
                // where delta[j,k] is 1 if j==k, 0 otherwise
                float softmax_k =
                    std::exp(x_data[row_start + k] - max_val) / sum_exp;
                float jacobian_jk =
                    softmax_j * ((j == k ? 1.0f : 0.0f) - softmax_k);
                grad_sum += jacobian_jk * output_grad_data[row_start + k];
            }
            x_grad_data[row_start + j] = grad_sum;
        }
    }
}

void softmax_backward(const Tensor &output_grad, const Tensor &output,
                      Tensor &x_grad) {
    size_t m = output.shape[0]; // Number of rows
    size_t n = output.shape[1]; // Number of columns

    parallel_for(m, softmax_backward_task, n, std::ref(output_grad.data),
                 std::ref(output.data), std::ref(x_grad.data));
}

// Function to apply Log backward pass
void log_backward_task(size_t start, size_t end,
                       const std::vector<float> &output_grad_data,
                       const std::vector<float> &x_data,
                       std::vector<float> &x_grad_data) {
    for (size_t i = start; i < end; ++i) {
        // Gradient of log(x) is 1/x, with a minimum value threshold
        float x = std::max(x_data[i], 1e-8f);
        x_grad_data[i] = output_grad_data[i] / x;
    }
}

void log_backward(const Tensor &output_grad, const Tensor &x, Tensor &x_grad) {
    parallel_for(x.data.size(), log_backward_task, std::ref(output_grad.data),
                 std::ref(x.data), std::ref(x_grad.data));
}

void sum_backward_task(size_t start, size_t end, float value,
                       std::vector<float> &x_grad_data) {
    for (size_t i = start; i < end; i++) {
        x_grad_data[i] = value;
    }
}

void sum_backward(const Tensor &output_grad, const Tensor &x, Tensor &x_grad) {
    parallel_for(x.data.size(), sum_backward_task, output_grad.data[0],
                 std::ref(x_grad.data));
}

}; // namespace ops
