#include "ops.h"
#include <algorithm>
#include <vector>
#include <thread>
#include <future>
#include <numeric>
#include <cmath>
#include <limits>

namespace py = pybind11;

template <typename Func, typename... Args>
void parallel_for(size_t total_work, Func&& func, Args&&... args) {
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures;

    size_t work_per_thread = total_work / num_threads;
    size_t remaining_work = total_work % num_threads;
    size_t start = 0;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t work = work_per_thread + (t < remaining_work ? 1 : 0);
        size_t end = start + work;

        if (work > 0) {
            futures.push_back(std::async(std::launch::async, func, start, end, std::forward<Args>(args)...));
        }
        start = end;
    }

    for (auto& future : futures) {
        future.get();
    }
}


// Function to perform matrix multiplication for a subset of rows
void matmul_task(size_t start_row, size_t end_row, size_t m, size_t n, size_t k,
                      const std::vector<float>& a_data, const std::vector<float>& b_data,
                      std::vector<float>& result_data) {
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; ++l) {
                sum += a_data[i * k + l] * b_data[l * n + j];
            }
            result_data[i * n + j] = sum;
        }
    }
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    const auto& a_shape = a.get_shape();
    const auto& b_shape = b.get_shape();

    size_t m = a_shape[0];
    size_t k = a_shape[1];
    size_t n = b_shape[1];

    Tensor result({m, n});
    parallel_for(m, matmul_task, m, n, k, a.get_data_vector(), b.get_data_vector(),
                 std::ref(const_cast<std::vector<float>&>(result.get_data_vector())));

    return result;
}

// Function to perform row-wise addition for a subset of rows
void add_broadcast_row_task(size_t start_row, size_t end_row, size_t n,
                                 const std::vector<float>& a_data, const std::vector<float>& b_data,
                                 std::vector<float>& result_data) {
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result_data[i * n + j] = a_data[i * n + j] + b_data[j];
        }
    }
}

Tensor add_broadcast_row(const Tensor& a, const Tensor& b) {
    const auto& a_shape = a.get_shape();
    const auto& b_shape = b.get_shape();

    size_t m = a_shape[0];
    size_t n = a_shape[1];

    Tensor result({m, n});

    parallel_for(m, add_broadcast_row_task, n, a.get_data_vector(), b.get_data_vector(),
                 std::ref(const_cast<std::vector<float>&>(result.get_data_vector())));

    return result;
}


// Function to apply ReLU to a subset of the tensor
void relu_task(size_t start_index, size_t end_index, std::vector<float>& data) {
    for (size_t i = start_index; i < end_index; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

Tensor relu(const Tensor& tensor) {
    Tensor result(tensor.get_shape());
    auto& result_data = const_cast<std::vector<float>&>(result.get_data_vector());
    std::copy(tensor.get_data_vector().begin(), tensor.get_data_vector().end(), result_data.begin());

    parallel_for(result_data.size(), relu_task, std::ref(result_data));
    return result;
}


// Function to apply Softmax to a subset of rows
void softmax_task(size_t start_row, size_t end_row, size_t n, std::vector<float>& data) {
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
            data[j] = std::exp(data[j] - max_val); // Subtract max_val for stability
            sum_exp += data[j];
        }

        // Normalize the row by dividing by the sum of exponentials
        for (size_t j = row_start; j < row_end; ++j) {
            data[j] /= sum_exp;
        }
    }
}

Tensor softmax(const Tensor& tensor) {
    const auto& shape = tensor.get_shape();
    size_t m = shape[0]; // Number of rows
    size_t n = shape[1]; // Number of columns

    Tensor result(shape);
    auto& result_data = const_cast<std::vector<float>&>(result.get_data_vector());
    std::copy(tensor.get_data_vector().begin(), tensor.get_data_vector().end(), result_data.begin());

    parallel_for(m, softmax_task, n, std::ref(result_data));

    return result;
}

void log_task(size_t start, size_t end, std::vector<float>& data) {
    for (size_t i = start; i < end; ++i) {
        data[i] = (data[i] > 0.0f) ? std::log(data[i]) : std::numeric_limits<float>::quiet_NaN();
    }
}

Tensor element_wise_log(const Tensor& tensor) {
    Tensor result(tensor.get_shape());
    auto& result_data = const_cast<std::vector<float>&>(result.get_data_vector());
    std::copy(tensor.get_data_vector().begin(), tensor.get_data_vector().end(), result_data.begin());

    parallel_for(result_data.size(), log_task, std::ref(result_data));
    return result;
}
