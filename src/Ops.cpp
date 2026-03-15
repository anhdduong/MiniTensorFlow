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

