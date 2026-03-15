#include "mt/engine.hpp"
#include <unordered_set>
#include <vector>

static void build_topo(std::shared_ptr<Node> node,
                        std::unordered_set<Node*>& visited,
                        std::vector<std::shared_ptr<Node>>& topo) {
    if (visited.count(node.get())) return;
    visited.insert(node.get());
    for (auto& parent : node->parents) {
        build_topo(parent, visited, topo);
    }
    topo.push_back(node);
}

void backward(std::shared_ptr<Node> root) {
    std::vector<std::shared_ptr<Node>> topo;
    std::unordered_set<Node*> visited;
    build_topo(root, visited, topo);

    root->grad.fill(1.0f);

    for (int i = topo.size() - 1; i >= 0; i--) {
        topo[i]->backward();
    }
}
