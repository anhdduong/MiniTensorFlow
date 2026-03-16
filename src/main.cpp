#include <iostream>
#include "mt/tensor.hpp"
#include "mt/node.hpp"
#include "mt/ops.hpp"
#include "mt/engine.hpp"

int main() {
    // A: {2, 3}, B: {3, 2} — mix of positive and negative values to test relu
    Tensor ta({2, 3});
    ta.fill({1.0f, -2.0f, 3.0f,
             -4.0f, 5.0f, -6.0f});

    Tensor tb({3, 2});
    tb.fill({1.0f,  2.0f,
             3.0f,  4.0f,
             5.0f,  6.0f});

    auto a = Node::make(ta);
    auto b = Node::make(tb);

    // loss = sum(relu(matmul(a, b)))
    auto c    = matmul(a, b);
    auto r    = relu(c);
    auto loss = sum(r);

    std::cout << "c->data (matmul, shape {2,2}):\n";
    c->data.print();

    std::cout << "\nr->data (after relu, shape {2,2}):\n";
    r->data.print();

    std::cout << "\nloss->data (scalar):\n";
    loss->data.print();

    backward(loss);

    std::cout << "\na->grad (shape {2,3}):\n";
    a->grad.print();

    std::cout << "\nb->grad (shape {3,2}):\n";
    b->grad.print();

    return 0;
}
