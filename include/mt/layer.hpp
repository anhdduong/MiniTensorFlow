#pragma once
#include <memory>
#include <vector>
#include "mt/node.hpp"

class Layer {
public:
    virtual std::shared_ptr<Node> forward(std::shared_ptr<Node> x) = 0;
    virtual std::vector<std::shared_ptr<Node>> parameters() = 0;
    virtual ~Layer() = default;
};

class Linear : public Layer {
public:
    std::shared_ptr<Node> W;
    std::shared_ptr<Node> b;

    Linear(int in_features, int out_features);
    std::shared_ptr<Node> forward(std::shared_ptr<Node> x) override;
    std::vector<std::shared_ptr<Node>> parameters() override;
};

class ReLU : public Layer {
public:
    std::shared_ptr<Node> forward(std::shared_ptr<Node> x) override;
    std::vector<std::shared_ptr<Node>> parameters() override { return {}; }
};

class Sigmoid : public Layer {
public:
    std::shared_ptr<Node> forward(std::shared_ptr<Node> x) override;
    std::vector<std::shared_ptr<Node>> parameters() override { return {}; }
};

class Sequential : public Layer {
public:
    std::vector<std::unique_ptr<Layer>> layers;

    Sequential(std::vector<std::unique_ptr<Layer>> layers);
    std::shared_ptr<Node> forward(std::shared_ptr<Node> x) override;
    std::vector<std::shared_ptr<Node>> parameters() override;
};
