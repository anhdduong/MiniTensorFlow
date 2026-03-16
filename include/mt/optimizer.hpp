#pragma once
#include <memory>
#include <vector>
#include "mt/node.hpp"

class SGD {
public:
    SGD(std::vector<std::shared_ptr<Node>> parameters, float lr);
    void step();
    void zero_grad();

private:
    std::vector<std::shared_ptr<Node>> params;
    float lr;
};
