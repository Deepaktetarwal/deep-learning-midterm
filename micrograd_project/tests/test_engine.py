import math
import torch
from micrograd.engine import Value

def test_simple_expression():
    # compute y = (2*x + 2 + x).relu() + (2*x + 2 + x) * x + ((2*x + 2 + x)**2).relu()
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    x_mg_val = x.data
    x_mg_grad = x.grad

    # Now do same with torch
    xt = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
    zt = 2 * xt + 2 + xt
    qt = torch.relu(zt) + zt * xt
    ht = torch.relu(zt * zt)
    yt = ht + qt + qt * xt
    yt.backward()
    x_t_grad = xt.grad.item()
    x_t_val = xt.detach().item()

    assert math.isclose(x_mg_val, x_t_val, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(x_mg_grad, x_t_grad, rel_tol=1e-6, abs_tol=1e-6)

def test_varied_ops():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    a_mg_grad = a.grad
    b_mg_grad = b.grad
    g_mg_val = g.data

    import torch
    at = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
    bt = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    c = at + bt
    d = at * bt + bt ** 3
    c = c + c + 1
    c = c + 1 + c + (-at)
    d = d + d * 2 + torch.relu(bt + at)
    d = d + 3 * d + torch.relu(bt - at)
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    a_t_grad = at.grad.item()
    b_t_grad = bt.grad.item()
    g_t_val = g.detach().item()

    assert math.isclose(g_mg_val, g_t_val, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(a_mg_grad, a_t_grad, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(b_mg_grad, b_t_grad, rel_tol=1e-6, abs_tol=1e-6)
