#include "mt/tensor.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

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