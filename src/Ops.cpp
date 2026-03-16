#include "mt/ops.hpp"

std::shared_ptr<Node> add(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
    Tensor result_data = a->data + b->data;
    auto result = Node::make(result_data);
    result->backward = [a,b,result]() {
        a->grad += result->grad;
        b->grad += result->grad;
    };
    result->parents = {a,b};
    return result;
}

std::shared_ptr<Node> mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
    Tensor result_data = a->data * b->data;
    auto result = Node::make(result_data);
    result->backward = [a,b,result]() {
        a->grad += result->grad * b->data;
        b->grad += result->grad * a->data;
    };
    result->parents = {a,b};
    return result;
}

std::shared_ptr<Node> matmul(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
    Tensor result_data = a->data.matmul(b->data);
    auto result = Node::make(result_data);
    result->backward = [a, b, result]() {
        a->grad += result->grad.matmul(b->data.transpose());
        b->grad += a->data.transpose().matmul(result->grad);
    };
    result->parents = {a, b};
    return result;
}

std::shared_ptr<Node> relu(std::shared_ptr<Node> a) {
    Tensor result_data = a->data.relu();
    auto result = Node::make(result_data);
    result->backward = [a, result]() {
        for (size_t i = 0; i < a->grad.data.size(); i++) {
            a->grad.data[i] += result->grad.data[i] * (a->data.data[i] > 0.0f ? 1.0f : 0.0f);
        }
    };
    result->parents = {a};
    return result;
}

std::shared_ptr<Node> sum(std::shared_ptr<Node> a) {
    Tensor result_data = a->data.sum();
    auto result = Node::make(result_data);
    result->backward = [a, result]() {
        for (size_t i = 0; i < a->grad.data.size(); i++) {
            a->grad.data[i] += result->grad.data[0];
        }
    };
    result->parents = {a};
    return result;
}

