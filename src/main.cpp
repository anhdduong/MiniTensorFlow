#include <iostream>
#include "mt/tensor.hpp"
#include "mt/node.hpp"
#include "mt/ops.hpp"
#include "mt/engine.hpp"
#include "mt/layer.hpp"
#include "mt/optimizer.hpp"

int main() {
    std::srand(42);

    // XOR data
    std::vector<std::vector<float>> xs = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
    };
    std::vector<float> ys = {0.0f, 1.0f, 1.0f, 0.0f};

    // Build network: Linear(2,4) -> ReLU -> Linear(4,1) -> Sigmoid
    std::vector<std::unique_ptr<Layer>> layer_list;
    layer_list.push_back(std::make_unique<Linear>(2, 8));
    layer_list.push_back(std::make_unique<ReLU>());
    layer_list.push_back(std::make_unique<Linear>(8, 1));
    layer_list.push_back(std::make_unique<Sigmoid>());
    Sequential net(std::move(layer_list));

    SGD optimizer(net.parameters(), 0.1f);

    for (int step = 1; step <= 1000; step++) {
        std::shared_ptr<Node> total_loss = nullptr;

        for (size_t i = 0; i < xs.size(); i++) {
            Tensor x_data({1, 2});
            x_data.fill(xs[i]);

            Tensor y_data({1, 1});
            y_data.fill({ys[i]});

            auto x = Node::make(x_data);
            auto pred = net.forward(x);
            auto loss = bce_loss(pred, Node::make(y_data));

            total_loss = (total_loss == nullptr) ? loss : add(total_loss, loss);
        }

        if (step % 100 == 0) {
            std::cout << "step " << step << " loss: " << total_loss->data.data[0] << "\n";
        }

        backward(total_loss);
        optimizer.step();
        optimizer.zero_grad();
    }

    std::cout << "\nfinal predictions (should be ~[0, 1, 1, 0]):\n";
    for (size_t i = 0; i < xs.size(); i++) {
        Tensor x_data({1, 2});
        x_data.fill(xs[i]);
        auto x = Node::make(x_data);
        auto pred = net.forward(x);
        std::cout << "[" << xs[i][0] << "," << xs[i][1] << "] -> " << pred->data.data[0] << "\n";
    }

    return 0;
}
