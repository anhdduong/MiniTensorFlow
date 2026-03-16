#pragma once
#include "mt/node.hpp"

std::shared_ptr<Node> add(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> matmul(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> relu(std::shared_ptr<Node> a);
std::shared_ptr<Node> sum(std::shared_ptr<Node> a);
std::shared_ptr<Node> transpose(std::shared_ptr<Node> a);
std::shared_ptr<Node> sigmoid(std::shared_ptr<Node> a);
std::shared_ptr<Node> log_op(std::shared_ptr<Node> a);
std::shared_ptr<Node> bce_loss(std::shared_ptr<Node> pred, std::shared_ptr<Node> target);