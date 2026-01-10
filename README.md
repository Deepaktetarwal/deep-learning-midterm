# deep-learning-midterm
Created for midterm report submission of Wid's 2025

The repository micrograd_project contains a small, original implementation of a scalar automatic differentiation engine (`Value`) and a tiny neural network library (`Neuron`, `Layer`, `MLP`) built on top of it. It is intended as a learning project (for coursework / interview prep) and reproduces the essential behaviour of educational projects like "micrograd" but implemented independently.

## Goals
- Implement reverse-mode autodiff (backprop) on a dynamically-constructed computation graph.
- Build a small MLP on top of the autograd engine that supports ReLU activations.
- Validate the correctness of gradients by comparing with PyTorch on a variety of expressions and small networks.

## Files
- `micrograd/engine.py` — `Value` class: scalar value, operator overloads, and `backward()` implementation.
- `micrograd/nn.py` — `Neuron`, `Layer`, `MLP` classes using `Value` nodes as parameters.
- `tests/test_engine.py` — unit tests comparing scalar expression gradients with PyTorch.
- `tests/test_nn.py` — unit tests comparing MLP parameter gradients with PyTorch.


## How to run
 Install test dependencies:
   pip install pytest torch

   then in micrograd_project directory run:
        python -m pytest -q
This will automatically run the test in test_engine.py and test_nn.py
