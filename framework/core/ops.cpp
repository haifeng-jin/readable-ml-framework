#include "ops.h"
#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <numeric>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace ops {

template <typename Func, typename... Args>
void parallel_for(size_t total_work, Func &&func, Args &&...args) {
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures;

    size_t work_per_thread = total_work / num_threads;
    size_t remaining_work = total_work % num_threads;
    size_t start = 0;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t work = work_per_thread + (t < remaining_work ? 1 : 0);
        size_t end = start + work;

        if (work > 0) {
            futures.push_back(std::async(std::launch::async, func, start, end,
                                         std::forward<Args>(args)...));
        }
        start = end;
    }

    for (auto &future : futures) {
        future.get();
    }
}

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
void add_task(size_t start_row, size_t end_row, size_t n,
              const std::vector<float> &a_data,
              const std::vector<float> &b_data,
              std::vector<float> &output_data) {
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < n; ++j) {
            output_data[i * n + j] = a_data[i * n + j] + b_data[j];
        }
    }
}

void add(const Tensor &a, const Tensor &b, Tensor &output) {
    size_t m = a.shape[0];
    size_t n = a.shape[1];

    parallel_for(m, add_task, n, std::ref(a.data), std::ref(b.data),
                 std::ref(output.data));
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
