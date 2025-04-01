#include "ops.h"
#include <stdexcept>
#include <vector>
#include <thread>
#include <future>

namespace py = pybind11;

// Function to perform matrix multiplication for a subset of rows
void matmul_thread_task(size_t start_row, size_t end_row, size_t m, size_t n, size_t k,
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
    std::vector<float>& result_data = const_cast<std::vector<float>&>(result.get_data_vector());

    const auto& a_data = a.get_data_vector();
    const auto& b_data = b.get_data_vector();

    // Determine a reasonable number of threads to use.
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures;

    // Determine the number of rows each thread will process
    size_t rows_per_thread = m / num_threads;
    size_t remaining_rows = m % num_threads;

    size_t start_row = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t num_rows = rows_per_thread + (t < remaining_rows ? 1 : 0);
        size_t end_row = start_row + num_rows;

        if (num_rows > 0) {
            // Use the named function matmul_thread_task
            futures.push_back(std::async(std::launch::async,
                                         matmul_thread_task, start_row, end_row, m, n, k,
                                         std::ref(a_data), std::ref(b_data), std::ref(result_data)));
        }
        start_row = end_row;
    }

    // Wait for all threads to complete
    for (auto& future : futures) {
        future.get();
    }

    return result;
}

// Function to perform row-wise addition for a subset of rows
void add_broadcast_row_thread_task(size_t start_row, size_t end_row, size_t n,
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
    std::vector<float>& result_data = const_cast<std::vector<float>&>(result.get_data_vector());

    const auto& a_data = a.get_data_vector();
    const auto& b_data = b.get_data_vector();

    // Determine a reasonable number of threads to use.
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures;

    // Determine the number of rows each thread will process
    size_t rows_per_thread = m / num_threads;
    size_t remaining_rows = m % num_threads;

    size_t start_row = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t num_rows = rows_per_thread + (t < remaining_rows ? 1 : 0);
        size_t end_row = start_row + num_rows;

        if (num_rows > 0) {
            // Use the named function add_broadcast_row_thread_task
            futures.push_back(std::async(std::launch::async,
                                         add_broadcast_row_thread_task, start_row, end_row, n,
                                         std::ref(a_data), std::ref(b_data), std::ref(result_data)));
        }
        start_row = end_row;
    }

    // Wait for all threads to complete
    for (auto& future : futures) {
        future.get();
    }

    return result;
}


// Function to apply ReLU to a subset of the tensor
void relu_thread_task(size_t start_index, size_t end_index, std::vector<float>& data) {
    for (size_t i = start_index; i < end_index; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

Tensor relu(const Tensor& tensor) {
    const auto& shape = tensor.get_shape();
    Tensor result(shape); // Create a new tensor with the same shape
    std::vector<float>& result_data = const_cast<std::vector<float>&>(result.get_data_vector());
    const std::vector<float>& data = tensor.get_data_vector();

    size_t num_elements = result_data.size();
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures;

    size_t elements_per_thread = num_elements / num_threads;
    size_t remaining_elements = num_elements % num_threads;
    size_t start_index = 0;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t num_elements_thread = elements_per_thread + (t < remaining_elements ? 1 : 0);
        size_t end_index = start_index + num_elements_thread;

        if (num_elements_thread > 0) {
            futures.push_back(std::async(std::launch::async,
                                         relu_thread_task, start_index, end_index, std::ref(result_data)));
        }
        start_index = end_index;
    }

    for (auto& future : futures) {
        future.get();
    }

    // Copy the data from the input tensor to the result tensor
    std::copy(data.begin(), data.end(), result_data.begin());
    // Apply relu in place.
    relu_thread_task(0, num_elements, result_data);

    return result;
}


// Function to apply Softmax to a subset of rows
void softmax_thread_task(size_t start_row, size_t end_row, size_t n, std::vector<float>& data) {
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
    if (shape.size() != 2) {
        throw std::runtime_error("Softmax requires a 2D tensor.");
    }
    size_t m = shape[0]; // Number of rows
    size_t n = shape[1]; // Number of columns

    Tensor result(shape);
    std::vector<float>& result_data = const_cast<std::vector<float>&>(result.get_data_vector());
    const std::vector<float>& data = tensor.get_data_vector();

    // Copy the data to the result tensor.  Softmax is applied in-place.
      std::copy(data.begin(), data.end(), result_data.begin());

    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures;

    size_t rows_per_thread = m / num_threads;
    size_t remaining_rows = m % num_threads;
    size_t start_row = 0;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t num_rows = rows_per_thread + (t < remaining_rows ? 1 : 0);
        size_t end_row = start_row + num_rows;
        if (num_rows > 0) {
          futures.push_back(std::async(std::launch::async, softmax_thread_task, start_row, end_row, n, std::ref(result_data)));
        }
        start_row = end_row;
    }
     for (auto& future : futures) {
        future.get();
    }
    return result;
}
