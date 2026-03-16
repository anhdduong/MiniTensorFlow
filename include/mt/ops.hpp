#pragma once
#include "mt/node.hpp"

std::shared_ptr<Node> add(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> matmul(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> relu(std::shared_ptr<Node> a);
std::shared_ptr<Node> sum(std::shared_ptr<Node> a);