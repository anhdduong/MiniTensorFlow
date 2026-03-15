#include <iostream>
#include "mt/tensor.hpp"
#include "mt/node.hpp"
#include "mt/ops.hpp"
#include "mt/engine.hpp"

int main() {
    Tensor t1({2, 3});
    t1.fill({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    Tensor t2({2, 3});
    t2.fill({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    auto a = Node::make(t1);
    auto b = Node::make(t2);
    // chain: a * b = c, c + d = e
    Tensor t3({2, 3});
    t3.fill(1.0f);
    auto d = Node::make(t3);

    auto c = mul(a, b);
    auto e = add(c, d);

    backward(e);

    std::cout << "e->data (forward):\n";
    e->data.print();

    std::cout << "a->grad (should equal b->data):\n";
    a->grad.print();

    std::cout << "b->grad (should equal a->data):\n";
    b->grad.print();

    std::cout << "d->grad (should be all ones):\n";
    d->grad.print();

    return 0;
}
