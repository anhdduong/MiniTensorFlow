#include "mt/optimizer.hpp"

SGD::SGD(std::vector<std::shared_ptr<Node>> parameters, float lr)
    : params(parameters), lr(lr) {}

void SGD::step() {
    for (auto& p : params) {
        p->data = p->data - p->grad * lr;
    }
}

void SGD::zero_grad() {
    for (auto& p : params) {
        p->grad.fill(0.0f);
    }
}
