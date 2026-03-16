#include "mt/layer.hpp"
#include "mt/ops.hpp"
#include <cstdlib>

// --- Linear ---

Linear::Linear(int in_features, int out_features) {
    Tensor w_data({out_features, in_features});
    for (size_t i = 0; i < w_data.data.size(); i++) {
        w_data.data[i] = ((float)std::rand() / RAND_MAX) * 1.0f - 0.5f;
    }
    W = Node::make(w_data);

    Tensor b_data({1, out_features});
    b_data.fill(0.0f);
    b = Node::make(b_data);
}

std::shared_ptr<Node> Linear::forward(std::shared_ptr<Node> x) {
    auto wt = transpose(W);
    auto out = matmul(x, wt);
    return add(out, b);
}

std::vector<std::shared_ptr<Node>> Linear::parameters() {
    return {W, b};
}

// --- ReLU ---

std::shared_ptr<Node> ReLU::forward(std::shared_ptr<Node> x) {
    return relu(x);
}

// --- Sigmoid ---

std::shared_ptr<Node> Sigmoid::forward(std::shared_ptr<Node> x) {
    return sigmoid(x);
}

// --- Sequential ---

Sequential::Sequential(std::vector<std::unique_ptr<Layer>> layers)
    : layers(std::move(layers)) {}

std::shared_ptr<Node> Sequential::forward(std::shared_ptr<Node> x) {
    auto out = x;
    for (auto& layer : layers) {
        out = layer->forward(out);
    }
    return out;
}

std::vector<std::shared_ptr<Node>> Sequential::parameters() {
    std::vector<std::shared_ptr<Node>> params;
    for (auto& layer : layers) {
        auto lp = layer->parameters();
        params.insert(params.end(), lp.begin(), lp.end());
    }
    return params;
}
