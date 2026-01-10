import random
from .engine import Value
from typing import List

random.seed(42)

class Neuron:
    def __init__(self, nin: int, nonlin: bool = True):
        # create weights and a bias as Value scalars
        self.w: List[Value] = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b: Value = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: List[Value]):
        # x: list of Value inputs (length == len(self.w))
        assert len(x) == len(self.w)
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi
        return act.relu() if self.nonlin else act

    def parameters(self):
        # all trainable Values: weights and bias
        return [p for p in self.w] + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer:
    def __init__(self, nin: int, nout: int, nonlin: bool = True):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x: List[Value]):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        # flatten parameters from all neurons
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return "Layer of [" + ", ".join(str(n) for n in self.neurons) + "]"

class MLP:
    def __init__(self, nin: int, nouts: List[int]):
        # sizes: nin -> nouts[0] -> nouts[1] -> ... -> nouts[-1]
        sizes = [nin] + nouts
        # non-linearity for all layers except last
        self.layers = [Layer(sizes[i], sizes[i+1], nonlin=(i != len(nouts)-1))
                       for i in range(len(nouts))]

    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return "MLP of [" + ", ".join(str(layer) for layer in self.layers) + "]"
