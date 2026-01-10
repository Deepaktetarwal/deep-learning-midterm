from typing import Iterable, Callable, Tuple
import math

class Value:


    def __init__(self, data: float, _children: Iterable["Value"]=(), _op: str=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        # placeholder; will be overwritten by operator implementations
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"


    def _coerce(self, other):
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self._coerce(other)
        out = Value(self.data + other.data, (self, other), _op="+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._coerce(other) if isinstance(other, Value) else -other)

    def __rsub__(self, other):
        return self._coerce(other) + (-self)

    def __mul__(self, other):
        other = self._coerce(other)
        out = Value(self.data * other.data, (self, other), _op="*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = self._coerce(other)
        return self * other**-1

    def __rtruediv__(self, other):
        other = self._coerce(other)
        return other * (self ** -1)

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "exponent must be constant number"
        out = Value(self.data ** exponent, (self,), _op=f"**{exponent}")
        def _backward():
            self.grad += (exponent * (self.data ** (exponent - 1))) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), _op="ReLU")
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def backward(self):

        topo = []
        visited = set()
        def build(v: "Value"):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build(parent)
                topo.append(v)
        build(self)

        self.grad = 1.0
        # go backward
        for node in reversed(topo):
            node._backward()
