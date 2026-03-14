#include <iostream>
#include "mt/tensor.hpp"
#include "mt/node.hpp"

int main() {
    Tensor t({2, 3});
    t.fill({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    auto node = Node::make(t);

    std::cout << "data:\n";
    node->data.print();

    std::cout << "grad:\n";
    node->grad.print();

    return 0;
}
