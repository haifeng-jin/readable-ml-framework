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

/*
 * @brief This function run tasks in parallel with multi-threading.
 *
 * @param total_work: The number of tasks that could run in parallel.
 * @param func: The function that runs an interval of tasks. The function
 * should accept two size_t type parameters, start and end, before the rest of
 * the arguments. Start and end marks the interval of tasks that the task
 * function should run.
 * @param args: The arguments to pass to the task function.
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

// Function to perform matrix multiplication for a subset of rows
void matmul_task(size_t start_row, size_t end_row, size_t m, size_t n, size_t k,
                 const std::vector<float> &a_data,
                 const std::vector<float> &b_data,
                 std::vector<float> &output_data) {
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; ++l) {
                sum += a_data[i * k + l] * b_data[l * n + j];
            }
            output_data[i * n + j] = sum;
        }
    }
}

void matmul(const Tensor &a, const Tensor &b, Tensor &output) {
    size_t m = a.shape[0];
    size_t k = a.shape[1];
    size_t n = b.shape[1];

    parallel_for(m, matmul_task, m, n, k, std::ref(a.data), std::ref(b.data),
                 std::ref(output.data));
}

// Function to perform row-wise addition for a subset of rows
void add_row_broadcast_task(size_t start_row, size_t end_row, size_t n,
                            const std::vector<float> &a_data,
                            const std::vector<float> &b_data,
                            std::vector<float> &output_data) {
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < n; ++j) {
            output_data[i * n + j] = a_data[i * n + j] + b_data[j];
        }
    }
}

void add_row_broadcast(const Tensor &a, const Tensor &b, Tensor &output) {
    size_t m = a.shape[0];
    size_t n = a.shape[1];

    parallel_for(m, add_row_broadcast_task, n, std::ref(a.data),
                 std::ref(b.data), std::ref(output.data));
}

void add_element_wise_task(size_t start, size_t end, std::vector<float> &a_data,
                           const std::vector<float> &b_data) {
    for (size_t i = start; i < end; ++i) {
        a_data[i] += b_data[i];
    }
}

void add_element_wise_(Tensor &a, const Tensor &b) {
    parallel_for(a.data.size(), add_element_wise_task, std::ref(a.data),
                 std::ref(b.data));
}

// Function to perform element-wise multiply for a subset of rows
void multiply_task(size_t start, size_t end, const std::vector<float> &b_data,
                   std::vector<float> &output_data) {
    for (size_t i = start; i < end; ++i) {
        output_data[i] *= b_data[i];
    }
}

void multiply(const Tensor &a, const Tensor &b, Tensor &output) {
    std::copy(a.data.begin(), a.data.end(), output.data.begin());

    parallel_for(output.data.size(), multiply_task, std::ref(b.data),
                 std::ref(output.data));
}

// Function to apply ReLU to a subset of the tensor
void relu_task(size_t start_index, size_t end_index, std::vector<float> &data) {
    for (size_t i = start_index; i < end_index; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

void relu(const Tensor &tensor, Tensor &output) {
    std::copy(tensor.data.begin(), tensor.data.end(), output.data.begin());

    parallel_for(output.data.size(), relu_task, std::ref(output.data));
}

// Function to apply Softmax to a subset of rows
void softmax_task(size_t start_row, size_t end_row, size_t n,
                  std::vector<float> &data) {
    for (size_t i = start_row; i < end_row; ++i) {
        size_t row_start = i * n;
        size_t row_end = row_start + n;
        float max_val = -std::numeric_limits<float>::infinity();

        // Find the maximum value in the row for numerical stability
        for (size_t j = row_start; j < row_end; ++j) {
            if (data[j] > max_val) {
                max_val = data[j];
            }
        }

        float sum_exp = 0.0f;
        // Calculate the exponential of each element and the sum of exponentials
        for (size_t j = row_start; j < row_end; ++j) {
            data[j] =
                std::exp(data[j] - max_val); // Subtract max_val for stability
            sum_exp += data[j];
        }

        // Normalize the row by dividing by the sum of exponentials
        for (size_t j = row_start; j < row_end; ++j) {
            data[j] /= sum_exp;
        }
    }
}

void softmax(const Tensor &tensor, Tensor &output) {
    size_t m = tensor.shape[0]; // Number of rows
    size_t n = tensor.shape[1]; // Number of columns

    std::copy(tensor.data.begin(), tensor.data.end(), output.data.begin());

    parallel_for(m, softmax_task, n, std::ref(output.data));
}

void log_task(size_t start, size_t end, std::vector<float> &data) {
    for (size_t i = start; i < end; ++i) {
        data[i] = (data[i] > 1e-8) ? std::log(data[i]) : std::log(1e-8);
    }
}

void log(const Tensor &tensor, Tensor &output) {
    std::copy(tensor.data.begin(), tensor.data.end(), output.data.begin());

    parallel_for(output.data.size(), log_task, std::ref(output.data));
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

void sum(const Tensor &tensor, Tensor &output) {
    std::vector<float> partial_sum(tensor.shape[0]);
    parallel_for(tensor.shape[0], sum_task, tensor.shape[1],
                 std::ref(tensor.data), std::ref(partial_sum));
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
                            const std::vector<float> &b_data,
                            std::vector<float> &a_grad_data) {
    // Computing gradient with respect to A: dL/dA = dL/dC * B^T
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < k; ++j) {
            float sum = 0.0f;
            for (size_t l = 0; l < n; ++l) {
                sum += output_grad_data[i * n + l] * b_data[j * n + l];
            }
            a_grad_data[i * k + j] = sum;
        }
    }
}

void matmul_backward_task_b(size_t start_col, size_t end_col, size_t m,
                            size_t n, size_t k,
                            const std::vector<float> &output_grad_data,
                            const std::vector<float> &a_data,
                            std::vector<float> &b_grad_data) {
    // Computing gradient with respect to B: dL/dB = A^T * dL/dC
    for (size_t j = start_col; j < end_col; ++j) {
        for (size_t i = 0; i < k; ++i) {
            float sum = 0.0f;
            for (size_t l = 0; l < m; ++l) {
                sum += a_data[l * k + i] * output_grad_data[l * n + j];
            }
            b_grad_data[i * n + j] = sum;
        }
    }
}

void matmul_backward(const Tensor &output_grad, const Tensor &a,
                     const Tensor &b, Tensor &a_grad, Tensor &b_grad) {
    size_t m = a.shape[0];
    size_t k = a.shape[1];
    size_t n = b.shape[1];

    // Calculate gradients for A and B in parallel
    parallel_for(m, matmul_backward_task_a, m, n, k, std::ref(output_grad.data),
                 std::ref(b.data), std::ref(a_grad.data));

    parallel_for(n, matmul_backward_task_b, m, n, k, std::ref(output_grad.data),
                 std::ref(a.data), std::ref(b_grad.data));
}

// Function to perform add backward pass for a subset of columns
void add_row_broadcast_backward_task_b(
    size_t start_col, size_t end_col, size_t m, size_t n,
    const std::vector<float> &output_grad_data,
    std::vector<float> &b_grad_data) {
    // For b_grad, we need to sum the gradients across all rows for each column
    for (size_t j = start_col; j < end_col; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < m; ++i) {
            sum += output_grad_data[i * n + j];
        }
        b_grad_data[j] = sum;
    }
}

void add_row_broadcast_backward(const Tensor &output_grad, const Tensor &a,
                                const Tensor &b, Tensor &a_grad,
                                Tensor &b_grad) {
    size_t m = a.shape[0];
    size_t n = a.shape[1];

    // For a_grad, simply copy the output gradient as it flows through unchanged
    std::copy(output_grad.data.begin(), output_grad.data.end(),
              a_grad.data.begin());

    // For b_grad, sum the gradients across each column
    parallel_for(n, add_row_broadcast_backward_task_b, m, n,
                 std::ref(output_grad.data), std::ref(b_grad.data));
}

// Function to perform element-wise multiply backward pass
void multiply_backward_task_a(size_t start, size_t end,
                              const std::vector<float> &output_grad_data,
                              const std::vector<float> &b_data,
                              std::vector<float> &a_grad_data) {
    for (size_t i = start; i < end; ++i) {
        a_grad_data[i] = output_grad_data[i] * b_data[i];
    }
}

void multiply_backward_task_b(size_t start, size_t end,
                              const std::vector<float> &output_grad_data,
                              const std::vector<float> &a_data,
                              std::vector<float> &b_grad_data) {
    for (size_t i = start; i < end; ++i) {
        b_grad_data[i] = output_grad_data[i] * a_data[i];
    }
}

void multiply_backward(const Tensor &output_grad, const Tensor &a,
                       const Tensor &b, Tensor &a_grad, Tensor &b_grad) {
    // For element-wise multiplication, gradients are calculated as:
    // dL/dA = dL/dOutput * B
    // dL/dB = dL/dOutput * A

    parallel_for(a.data.size(), multiply_backward_task_a,
                 std::ref(output_grad.data), std::ref(b.data),
                 std::ref(a_grad.data));

    parallel_for(b.data.size(), multiply_backward_task_b,
                 std::ref(output_grad.data), std::ref(a.data),
                 std::ref(b_grad.data));
}

// Function to apply ReLU backward pass for a subset of the tensor
void relu_backward_task(size_t start, size_t end,
                        const std::vector<float> &output_grad_data,
                        const std::vector<float> &input_data,
                        std::vector<float> &input_grad_data) {
    for (size_t i = start; i < end; ++i) {
        // Gradient is output_grad if input was positive, 0 otherwise
        input_grad_data[i] = input_data[i] > 0 ? output_grad_data[i] : 0.0f;
    }
}

void relu_backward(const Tensor &output_grad, const Tensor &input,
                   Tensor &input_grad) {
    parallel_for(input.data.size(), relu_backward_task,
                 std::ref(output_grad.data), std::ref(input.data),
                 std::ref(input_grad.data));
}

// Function to apply Softmax backward pass for a subset of rows
void softmax_backward_task(size_t start_row, size_t end_row, size_t n,
                           const std::vector<float> &output_grad_data,
                           const std::vector<float> &input_data,
                           std::vector<float> &input_grad_data) {
    for (size_t i = start_row; i < end_row; ++i) {
        size_t row_start = i * n;
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < n; ++j) {
            if (input_data[row_start + j] > max_val) {
                max_val = input_data[row_start + j];
            }
        }

        float sum_exp = 0.0f;
        // Calculate the exponential of each element and the sum of exponentials
        for (size_t j = 0; j < n; ++j) {
            sum_exp += std::exp(input_data[row_start + j] -
                                max_val); // Subtract max_val for stability
        }

        // For each row, calculate the Jacobian vector product
        for (size_t j = 0; j < n; ++j) {
            float grad_sum = 0.0f;
            float softmax_j =
                std::exp(input_data[row_start + j] - max_val) / sum_exp;
            for (size_t k = 0; k < n; ++k) {
                // Jacobian of softmax: J[j,k] = s[j]*(delta[j,k] - s[k])
                // where delta[j,k] is 1 if j==k, 0 otherwise
                float softmax_k =
                    std::exp(input_data[row_start + k] - max_val) / sum_exp;
                float jacobian_jk =
                    softmax_j * ((j == k ? 1.0f : 0.0f) - softmax_k);
                grad_sum += jacobian_jk * output_grad_data[row_start + k];
            }
            input_grad_data[row_start + j] = grad_sum;
        }
    }
}

void softmax_backward(const Tensor &output_grad, const Tensor &output,
                      Tensor &input_grad) {
    size_t m = output.shape[0]; // Number of rows
    size_t n = output.shape[1]; // Number of columns

    parallel_for(m, softmax_backward_task, n, std::ref(output_grad.data),
                 std::ref(output.data), std::ref(input_grad.data));
}

// Function to apply Log backward pass
void log_backward_task(size_t start, size_t end,
                       const std::vector<float> &output_grad_data,
                       const std::vector<float> &input_data,
                       std::vector<float> &input_grad_data) {
    for (size_t i = start; i < end; ++i) {
        // Gradient of log(x) is 1/x, with a minimum value threshold
        float x = std::max(input_data[i], 1e-8f);
        input_grad_data[i] = output_grad_data[i] / x;
    }
}

void log_backward(const Tensor &output_grad, const Tensor &input,
                  Tensor &input_grad) {
    parallel_for(input.data.size(), log_backward_task,
                 std::ref(output_grad.data), std::ref(input.data),
                 std::ref(input_grad.data));
}

void sum_backward_task(size_t start, size_t end, float value,
                       std::vector<float> &input_grad_data) {
    for (size_t i = start; i < end; i++) {
        input_grad_data[i] = value;
    }
}

void sum_backward(const Tensor &output_grad, const Tensor &tensor,
                  Tensor &input_grad) {
    parallel_for(tensor.data.size(), sum_backward_task, output_grad.data[0],
                 std::ref(input_grad.data));
}

}; // namespace ops
