#include "mt/node.hpp"

Node::Node(Tensor tensor) : data(tensor), grad(tensor.shape), backward([](){}) {
    grad.fill(0.0f);
}
