import math
import random
import torch
import numpy as np
from micrograd.nn import MLP
from micrograd.engine import Value

random.seed(1337)
torch.manual_seed(1337)

def tiny_dataset():
    # simple XOR-like tiny dataset but linear separable for test
    X = np.array([[0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]], dtype=np.float64)
    y = np.array([ -1.0, 1.0, 1.0, -1.0 ], dtype=np.float64)  # labels
    return X, y

def test_mlp_gradients():
    X, y = tiny_dataset()
    # create a tiny MLP 2 -> 4 -> 1
    model = MLP(2, [4,1])

    # choose a single datapoint to test gradient; convert inputs to Value
    xi = [Value(X[1,0]), Value(X[1,1])]
    out = model(xi)  # Value
    # hinge-like loss: L = max(0, 1 - y * score)
    target = y[1]
    loss = (Value(1.0) + (-target) * out).relu()  
    loss.backward()

    # get gradient vector from our model parameters
    mg_grads = [p.grad for p in model.parameters()]

    # Now build same network in PyTorch with same numeric params
    # copy params from our model into torch tensors
    params = model.parameters()
    # flatten and build weight tensors per layer shape
    # helper to extract shapes
    sizes = [2] + [4, 1]
    torch_params = []
    idx = 0
    for i in range(len(sizes)-1):
        nin = sizes[i]; nout = sizes[i+1]
        # reconstruct weights and biases from flat parameter list
        W = torch.zeros((nout, nin), dtype=torch.float64)
        b = torch.zeros((nout,), dtype=torch.float64)
        for j in range(nout):
            for k in range(nin):
                W[j,k] = params[idx].data; idx += 1
            b[j] = params[idx].data; idx += 1
        W.requires_grad_(); b.requires_grad_()
        torch_params.append((W,b))

    # now forward in torch
    xt = torch.tensor(X[1], dtype=torch.float64).requires_grad_(False)
    out_t = xt
    for i, (W,b) in enumerate(torch_params):
        out_t = out_t @ W.t() + b  # (nout,)
        if i != len(torch_params)-1:
            out_t = torch.relu(out_t)
    score = out_t.squeeze()
    loss_t = torch.relu(1.0 - target * score)
    loss_t.backward()

    # compare grads for each corresponding param
    # Flatten torch grads in same order
    t_grads = []
    for W,b in torch_params:
        for j in range(W.shape[0]):
            for k in range(W.shape[1]):
                t_grads.append(W.grad[j,k].item())
            t_grads.append(b.grad[j].item())

    # compare first N parameters
    for mg, tg in zip(mg_grads, t_grads):
        assert math.isclose(mg, tg, rel_tol=1e-6, abs_tol=1e-6)
