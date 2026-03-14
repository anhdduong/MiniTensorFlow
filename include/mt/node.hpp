#pragma once
#include "mt/tensor.hpp"
#include <memory>
#include <functional>

class Node {
    public:
        Tensor data;
        Tensor grad;
        std::vector<std::shared_ptr<Node>> parents;
        std::function<void()> backward;

        Node(Tensor tensor);
        static std::shared_ptr<Node> make(Tensor tensor) {
            return std::make_shared<Node>(tensor);
        }
};