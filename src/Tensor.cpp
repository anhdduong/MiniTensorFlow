#include "mt/tensor.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>

Tensor::Tensor(std::vector<int> shape) {
    this->shape = shape;
    size_t length_shape = shape.size();
    this->strides.resize(length_shape);
    int suffixProduct = 1;
    for (int i = length_shape - 1; i >= 0; i--) {
        strides[i] = suffixProduct;
        suffixProduct = suffixProduct * shape[i];
    }
    this->data.resize(suffixProduct);

}

float& Tensor::at(std::vector<int> indices) {
    int index = compute_flat_index(indices);
    return data.at(index);
}

const float& Tensor::at(std::vector<int> indices) const {
    int index = compute_flat_index(indices);
    return data.at(index);
}

int Tensor::compute_flat_index(std::vector<int> indices) const {
    if (indices.size() != shape.size()) {
        throw std::out_of_range("Mismatch size shape vs indices");
    }
    int index = 0;
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            throw std::out_of_range("Dimension " + std::to_string(i) + " is out of bounds");
        }
        index += indices[i] * strides[i];
    }

    return index;
}

void Tensor::print_recursive(size_t dimension, int current_flat_offset, int indent) const {
    std::string pad(indent, ' ');
    if (dimension == shape.size() - 1) {
        int cols_num = shape[dimension];
        for (int i = 0; i < cols_num; i++) {
            if (i == cols_num - 1) {
                std::cout << std::to_string(data.at(current_flat_offset + i));
            } else {
                std::cout << std::to_string(data.at(current_flat_offset + i)) + ", ";
            }
        }
    } else {
        int rows_num = shape[dimension];
        for (int i = 0; i < rows_num; i++) {
            if (i > 0) std::cout << pad;
            std::cout << "[";
            print_recursive(dimension + 1, current_flat_offset + i * strides[dimension], indent + 1);
            if (i == rows_num - 1) {
                std::cout << "]";
            } else {
                std::cout << "],\n";
            }
        }
    }
}

void Tensor::print() const {
    std::cout << "[";
    print_recursive(0, 0, 1);
    std::cout << "]\n";
}

void Tensor::fill(float value) {
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = value;
    }
}

void Tensor::fill(std::vector<float> values) {
    if (values.size() != data.size()) {
        throw std::invalid_argument("values size does not match data size.");
    }

    for (size_t i = 0; i < data.size(); i++) {
        data[i] = values[i];
    }
}

Tensor Tensor::operator+(const Tensor& other) const{
    // 1. Check shape match
    if (other.shape != this->shape) {
        throw std::invalid_argument("Shape mismatch between this and other.");
    }

    // 2. Create a new Tensor object
    Tensor result = Tensor(this->shape);

    // 3. Loop through both data of this and other to add and fill in this new Tensor object
    for (size_t i = 0; i < this->data.size(); i++) {
        result.data[i] = other.data[i] + this->data[i];
    }

    // 4. Return the Tensor
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    // 1. Check shape match
    if (other.shape != this->shape) {
        throw std::invalid_argument("Shape mismatch between this and other.");
    }

    // 2. Loop through this's data and in-place with the other's data
    for (size_t i = 0; i < this->data.size(); i++) {
        this->data[i] += other.data[i];
    }

    return *this;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (other.shape != this->shape) {
        throw std::invalid_argument("Shape mismatch between this and other.");
    }
    Tensor result(this->shape);
    for (size_t i = 0; i < this->data.size(); i++) {
        result.data[i] = this->data[i] - other.data[i];
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(this->shape);
    for (size_t i = 0; i < this->data.size(); i++) {
        result.data[i] = this->data[i] * scalar;
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (other.shape != this->shape) {
        throw std::invalid_argument("Shape mismatch between this and other.");
    }
    Tensor result(this->shape);
    for (size_t i = 0; i < this->data.size(); i++) {
        result.data[i] = this->data[i] * other.data[i];
    }
    return result;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    if (other.shape != this->shape) {
        throw std::invalid_argument("Shape mismatch between this and other.");
    }
    for (size_t i = 0; i < this->data.size(); i++) {
        this->data[i] *= other.data[i];
    }
    return *this;
}

Tensor Tensor::relu() const {
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] > 0.0f ? data[i] : 0.0f;
    }
    return result;
}

Tensor Tensor::sum() const {
    Tensor result({1});
    float total = 0.0f;
    for (size_t i = 0; i < data.size(); i++) {
        total += data[i];
    }
    result.data[0] = total;
    return result;
}

Tensor Tensor::sigmoid() const {
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
    return result;
}

Tensor Tensor::log_t() const {
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = std::log(data[i]);
    }
    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape.size() != 2 || other.shape.size() != 2) {
        throw std::invalid_argument("matmul only supports 2D tensors");
    }
    int m = shape[0];
    int k = shape[1];
    if (other.shape[0] != k) {
        throw std::invalid_argument("matmul inner dimensions must match");
    }
    int n = other.shape[1];
    Tensor result({m, n});
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += at({i, l}) * other.at({l, j});
            }
            result.at({i, j}) = sum;
        }
    }
    return result;
}

Tensor Tensor::transpose() const {
    if (shape.size() != 2) {
        throw std::invalid_argument("transpose only supports 2D tensors");
    }
    int m = shape[0];
    int n = shape[1];
    Tensor result({n, m});
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result.at({i, j}) = at({j, i});
        }
    }
    return result;
}