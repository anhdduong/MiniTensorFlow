# MiniTensorFlow

- This is a mini TensorFlow built from scratch in C++ with automatic differentiation, distributed training emulator, and a focus on learning about ML training/inference framework.

- MiniTensorFlow is a deep learning framework written in modern C++ 17. Every components in this project: tensors, autograd, ops, layers, optimizers, and distributed learning is implemented from scratch with no external ML dependencies. The two ultimate goals of this project are to understand how frameworks like TensorFlow and PyTorch work in real applications, and push toward real systems-level distributed training.

---
### Objectives
- Build a complete deep learning framework in C++ without relying on external ML libraries
- Implement autograd from scratch
- Design two distributed training system styles: parameter server and Ring AllReduce
- Benchmark distributed vs single-threaded training to measure real speedup

---
 
## Architecture
 
```
Tensor          raw multi-dimensional array, row-major storage, stride arithmetic
   ↓
Node            autograd graph node which holds value, gradient, parents, backward closure
   ↓
Ops             differentiable operations that forward math + backward gradient formulas
   ↓
Engine          topological sort + reverse traversal to run full backprop automatically
   ↓
Layer           composable building blocks (Linear, Sequential) with learnable parameters
   ↓
Optimizer       SGD, Adam: reads gradients, updates parameters, zeros grads
   ↓
Distributed     parallel training across threads/processes with gradient synchronization
```
