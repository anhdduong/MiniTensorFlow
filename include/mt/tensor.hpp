#pragma once
#include <vector>
struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<int> strides;

    Tensor(std::vector<int> shape);
    float& at(std::vector<int> indices);
    const float& at(std::vector<int> indices) const;
    void print() const;
    void fill(float value);
    void fill(std::vector<float> values);
    Tensor operator+(const Tensor& other) const;
    Tensor& operator+=(const Tensor& other);
    Tensor operator*(const Tensor& other) const;
    Tensor& operator*=(const Tensor& other);

    private:
        int compute_flat_index(std::vector<int> indices) const;
        void print_recursive(size_t dimension, int current_flat_offset, int indent) const;
};