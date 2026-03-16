#include "mt/ops.hpp"
#include <cmath>

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

std::shared_ptr<Node> transpose(std::shared_ptr<Node> a) {
    auto result = Node::make(a->data.transpose());
    result->backward = [a, result]() {
        a->grad += result->grad.transpose();
    };
    result->parents = {a};
    return result;
}

std::shared_ptr<Node> sigmoid(std::shared_ptr<Node> a) {
    Tensor out_data = a->data.sigmoid();
    auto result = Node::make(out_data);
    result->backward = [a, result]() {
        for (size_t i = 0; i < a->grad.data.size(); i++) {
            float s = result->data.data[i];
            a->grad.data[i] += result->grad.data[i] * s * (1.0f - s);
        }
    };
    result->parents = {a};
    return result;
}

std::shared_ptr<Node> log_op(std::shared_ptr<Node> a) {
    const float eps = 1e-7f;
    Tensor out_data(a->data.shape);
    for (size_t i = 0; i < a->data.data.size(); i++) {
        out_data.data[i] = std::log(a->data.data[i] + eps);
    }
    auto result = Node::make(out_data);
    result->backward = [a, result, eps]() {
        for (size_t i = 0; i < a->grad.data.size(); i++) {
            a->grad.data[i] += result->grad.data[i] / (a->data.data[i] + eps);
        }
    };
    result->parents = {a};
    return result;
}

std::shared_ptr<Node> bce_loss(std::shared_ptr<Node> pred, std::shared_ptr<Node> target) {
    // -y*log(p + eps) - (1-y)*log(1-p + eps), summed
    const float eps = 1e-7f;
    Tensor loss_data({1});
    float total = 0.0f;
    for (size_t i = 0; i < pred->data.data.size(); i++) {
        float p = pred->data.data[i];
        float y = target->data.data[i];
        total += -y * std::log(p + eps) - (1.0f - y) * std::log(1.0f - p + eps);
    }
    loss_data.data[0] = total;
    auto result = Node::make(loss_data);
    result->backward = [pred, target, result, eps]() {
        float g = result->grad.data[0];
        for (size_t i = 0; i < pred->grad.data.size(); i++) {
            float p = pred->data.data[i];
            float y = target->data.data[i];
            pred->grad.data[i] += g * (-y / (p + eps) + (1.0f - y) / (1.0f - p + eps));
        }
    };
    result->parents = {pred, target};
    return result;
}

