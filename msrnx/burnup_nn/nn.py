"""Dependency-free neural-network engine (pure NumPy) for the burnup framework.

* ``MLP``        fully connected network, analytic back-propagation (weights and
                 inputs), optional residual connections, portable state dicts.
* ``AdamW``      Adam with decoupled weight decay (mini-batch training).
* ``fit_adam``   mini-batch loop with plateau learning-rate decay, early stopping
                 on a validation objective and best-weight restore.
* ``fit_lbfgs``  full-batch L-BFGS (SciPy) with validation-based early stopping:
                 reaches far lower losses than Adam on small / medium data sets
                 (typical for depletion studies: 10^3 - 10^5 rows).
* ``gradient_check``  finite-difference verification of the back-propagation.

The training loops are generic: they take ``objective(net, idx) -> (loss, grads)``
(``idx`` = row indices of the mini-batch, ``None`` = all rows) so the burnup model
can add masks, sample weights and physics penalties without touching this file.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Callable, Optional, Sequence

import numpy as np

_GELU_C = math.sqrt(2.0 / math.pi)


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))          # overflow-free logistic


def _act_forward(name: str, z: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return np.tanh(z)
    if name == "silu":
        return z * _sigmoid(z)
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "softplus":
        return np.logaddexp(0.0, z)
    if name == "gelu":
        return 0.5 * z * (1.0 + np.tanh(_GELU_C * (z + 0.044715 * z * z * z)))
    if name == "linear":
        return z
    raise ValueError(f"unknown activation '{name}' (use {', '.join(ACTIVATIONS)})")


def _act_derivative(name: str, z: np.ndarray, g: np.ndarray) -> np.ndarray:
    """d act / dz given the pre-activation z and the activation value g."""
    if name == "tanh":
        return 1.0 - g * g
    if name == "silu":
        s = _sigmoid(z)
        return s + g * (1.0 - s)
    if name == "relu":
        return (z > 0.0).astype(z.dtype)
    if name == "softplus":
        return _sigmoid(z)
    if name == "gelu":
        t = np.tanh(_GELU_C * (z + 0.044715 * z * z * z))
        return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * _GELU_C * (1.0 + 3 * 0.044715 * z * z)
    if name == "linear":
        return np.ones_like(z)
    raise ValueError(f"unknown activation '{name}'")


ACTIVATIONS = ("tanh", "silu", "relu", "softplus", "gelu", "linear")


class MLP:
    """Fully connected network ``n_in -> hidden... -> n_out`` with a linear output.

    Parameters are stored as ``[W0, b0, W1, b1, ...]`` (``W`` has shape
    ``(fan_in, fan_out)``).  With ``residual=True`` every hidden layer whose input
    and output widths match becomes ``h = h_in + act(h_in W + b)``.
    """

    def __init__(self, n_in: int, n_out: int, hidden: Sequence[int] = (64, 64, 64),
                 activation: str = "silu", residual: bool = False, seed: int = 0):
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation '{activation}' (use {', '.join(ACTIVATIONS)})")
        self.sizes = [int(n_in), *[int(h) for h in hidden], int(n_out)]
        self.activation = activation
        self.residual = bool(residual)
        rng = np.random.default_rng(seed)
        self.params = []
        for a, b in zip(self.sizes[:-1], self.sizes[1:]):
            std = math.sqrt(2.0 / a) if activation == "relu" else math.sqrt(2.0 / (a + b))
            self.params.append(rng.normal(0.0, std, (a, b)))
            self.params.append(np.zeros(b))

    # ------------------------------------------------------------------ basics
    @property
    def n_layers(self) -> int:
        return len(self.sizes) - 1

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))

    def weights(self):
        """The weight matrices (biases excluded): targets of L2 / weight decay."""
        return self.params[0::2]

    def get_flat(self) -> np.ndarray:
        return np.concatenate([p.ravel() for p in self.params])

    def set_flat(self, x: np.ndarray):
        i = 0
        for k, p in enumerate(self.params):
            n = p.size
            self.params[k] = np.asarray(x[i:i + n], float).reshape(p.shape).copy()
            i += n

    def copy(self) -> "MLP":
        return copy.deepcopy(self)

    def state(self) -> dict:
        return {"sizes": list(self.sizes), "activation": self.activation,
                "residual": self.residual}

    @classmethod
    def from_state(cls, state: dict, params: Sequence[np.ndarray]) -> "MLP":
        s = state["sizes"]
        net = cls(s[0], s[-1], s[1:-1], state["activation"], state.get("residual", False))
        if len(params) != len(net.params):
            raise ValueError("parameter list does not match the network architecture")
        for k, p in enumerate(params):
            if np.shape(p) != net.params[k].shape:
                raise ValueError(f"parameter {k}: shape {np.shape(p)} != {net.params[k].shape}")
            net.params[k] = np.array(p, float)
        return net

    # ------------------------------------------------------------ forward/back
    def forward(self, X: np.ndarray, keep: bool = False):
        """Network output; with ``keep=True`` also the cache for ``backward``."""
        a = np.asarray(X, float)
        cache = []
        for l in range(self.n_layers - 1):
            W, b = self.params[2 * l], self.params[2 * l + 1]
            z = a @ W + b
            g = _act_forward(self.activation, z)
            h = g + a if (self.residual and W.shape[0] == W.shape[1]) else g
            if keep:
                cache.append((a, z, g))
            a = h
        y = a @ self.params[-2] + self.params[-1]
        if keep:
            cache.append(a)
            return y, cache
        return y

    __call__ = forward

    def backward(self, cache, dY: np.ndarray, input_grad: bool = False):
        """Gradients of a scalar loss given ``dY = dL/dY``.

        Returns the list of parameter gradients (same order as ``params``) and,
        with ``input_grad=True``, also ``dL/dX``.
        """
        grads = [None] * len(self.params)
        a_last = cache[-1]
        grads[-2] = a_last.T @ dY
        grads[-1] = dY.sum(axis=0)
        da = dY @ self.params[-2].T
        for l in range(self.n_layers - 2, -1, -1):
            a_in, z, g = cache[l]
            W = self.params[2 * l]
            dz = da * _act_derivative(self.activation, z, g)
            grads[2 * l] = a_in.T @ dz
            grads[2 * l + 1] = dz.sum(axis=0)
            if l > 0 or input_grad:
                da_in = dz @ W.T
                if self.residual and W.shape[0] == W.shape[1]:
                    da_in = da_in + da
                da = da_in
        if input_grad:
            return grads, da
        return grads

    def input_gradient(self, X: np.ndarray, output_index: int) -> np.ndarray:
        """d y[:, output_index] / d X for every row of X."""
        Y, cache = self.forward(X, keep=True)
        dY = np.zeros_like(Y)
        dY[:, output_index] = 1.0
        return self.backward(cache, dY, input_grad=True)[1]


# ---------------------------------------------------------------------------
# Optimizers and training loops
# ---------------------------------------------------------------------------
class AdamW:
    """Adam with decoupled weight decay applied to weight matrices only."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads, lr: Optional[float] = None, frozen=()):
        lr = self.lr if lr is None else lr
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for k, (p, g) in enumerate(zip(params, grads)):
            if k in frozen:
                continue
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * g * g
            upd = (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)
            if self.wd > 0 and k % 2 == 0:
                upd = upd + self.wd * p
            params[k] = p - lr * upd


def _clip(grads, max_norm: float):
    if not max_norm:
        return grads
    norm = math.sqrt(sum(float((g * g).sum()) for g in grads))
    if norm > max_norm:
        f = max_norm / (norm + 1e-12)
        return [g * f for g in grads]
    return grads


def fit_adam(net: MLP, objective: Callable, n_rows: int, val_fn: Optional[Callable] = None,
             epochs: int = 2000, batch_size: int = 256, lr: float = 2e-3,
             weight_decay: float = 0.0, lr_patience: int = 60, lr_factor: float = 0.5,
             min_lr: float = 1e-5, patience: int = 200, clip: float = 10.0, seed: int = 0,
             frozen=(), verbose: bool = False) -> dict:
    """Mini-batch AdamW with plateau decay, early stopping and best-weight restore."""
    rng = np.random.default_rng(seed)
    opt = AdamW(net.params, lr, weight_decay=weight_decay)
    hist = {"iter": [], "train": [], "val": [], "lr": [], "optimizer": "adam"}
    best, best_params, wait, lr_wait, cur_lr = np.inf, [p.copy() for p in net.params], 0, 0, lr
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(n_rows)
        tot = 0.0
        for s in range(0, n_rows, batch_size):
            idx = perm[s:s + batch_size]
            loss, grads = objective(net, idx)
            opt.step(net.params, _clip(grads, clip), cur_lr, frozen)
            tot += loss * len(idx)
        train = tot / n_rows
        val = val_fn(net) if val_fn is not None else train
        hist["iter"].append(ep + 1)
        hist["train"].append(float(train))
        hist["val"].append(float(val))
        hist["lr"].append(cur_lr)
        if val < best * (1.0 - 1e-4):
            best, best_params, wait, lr_wait = val, [p.copy() for p in net.params], 0, 0
        else:
            wait += 1
            lr_wait += 1
        if lr_wait >= lr_patience:
            cur_lr *= lr_factor
            lr_wait = 0
        if verbose and (ep % 100 == 0 or ep == epochs - 1):
            print(f"    epoch {ep + 1:5d}  train {train:.3e}  val {val:.3e}  lr {cur_lr:.1e}")
        if wait >= patience or cur_lr < min_lr:
            break
    net.params = best_params
    hist["best_val"] = float(best)
    hist["wall_s"] = time.time() - t0
    return hist


def fit_lbfgs(net: MLP, objective: Callable, val_fn: Optional[Callable] = None,
              max_iter: int = 3000, patience: int = 40, check_every: int = 10,
              frozen=(), verbose: bool = False, memory: int = 30) -> dict:
    """Full-batch L-BFGS with validation monitoring (best-validation restore).

    ``patience`` counts validation checks (every ``check_every`` iterations)
    without improvement before stopping.  Parameters listed in ``frozen``
    (indices into ``net.params``) are held fixed; ``memory`` is the number of
    stored curvature pairs.
    """
    from scipy.optimize import minimize

    sizes = [p.size for p in net.params]
    mask = np.concatenate([np.full(s, k not in frozen) for k, s in enumerate(sizes)])
    base = net.get_flat()
    hist = {"iter": [], "train": [], "val": [], "lr": [], "optimizer": "lbfgs"}
    state = {"it": 0, "best": np.inf, "best_x": base[mask].copy(), "wait": 0, "loss": np.inf}

    def unpack(x):
        full = base.copy()
        full[mask] = x
        net.set_flat(full)

    def fun(x):
        unpack(x)
        loss, grads = objective(net, None)
        state["loss"] = loss
        g = np.concatenate([gr.ravel() for gr in grads])[mask]
        return float(loss), g

    def callback(intermediate_result):
        state["it"] += 1
        it = state["it"]
        if it % check_every and it != 1:
            return
        x = intermediate_result.x
        unpack(x)
        val = val_fn(net) if val_fn is not None else float(intermediate_result.fun)
        train = float(intermediate_result.fun)
        hist["iter"].append(it)
        hist["train"].append(train)
        hist["val"].append(float(val))
        hist["lr"].append(float("nan"))
        if verbose and it % (check_every * 20) == 0:
            print(f"    iter {it:6d}  train {train:.3e}  val {val:.3e}")
        if val < state["best"] * (1.0 - 1e-4):
            state["best"], state["best_x"], state["wait"] = val, x.copy(), 0
        else:
            state["wait"] += 1
            if state["wait"] >= patience:
                raise StopIteration

    t0 = time.time()
    x0 = base[mask].copy()
    res = minimize(fun, x0, jac=True, method="L-BFGS-B", callback=callback,
                   options={"maxiter": max_iter, "maxfun": 3 * max_iter, "maxcor": memory,
                            "ftol": 1e-16, "gtol": 1e-12})
    # final state may be better than the last check
    unpack(res.x)
    val_end = val_fn(net) if val_fn is not None else float(res.fun)
    if val_end < state["best"]:
        state["best"], state["best_x"] = val_end, res.x.copy()
    unpack(state["best_x"])
    hist["best_val"] = float(state["best"])
    hist["wall_s"] = time.time() - t0
    hist["n_iter"] = int(state["it"])
    return hist


# ---------------------------------------------------------------------------
def gradient_check(net: MLP, X: np.ndarray, Y: np.ndarray, eps: float = 1e-6,
                   n_probe: int = 60, seed: int = 0) -> float:
    """Max relative error between analytic and central-difference gradients
    of ``0.5 * mean((net(X) - Y)^2)`` (parameters and inputs)."""
    rng = np.random.default_rng(seed)
    n = len(X)

    def loss_of(net_, X_):
        return 0.5 * float(((net_.forward(X_) - Y) ** 2).sum()) / n

    P, cache = net.forward(X, keep=True)
    grads, dX = net.backward(cache, (P - Y) / n, input_grad=True)
    flat_g = np.concatenate([g.ravel() for g in grads])
    x0 = net.get_flat()
    worst = 0.0
    for i in rng.choice(len(x0), min(n_probe, len(x0)), replace=False):
        xp, xm = x0.copy(), x0.copy()
        xp[i] += eps
        xm[i] -= eps
        net.set_flat(xp)
        lp = loss_of(net, X)
        net.set_flat(xm)
        lm = loss_of(net, X)
        num = (lp - lm) / (2 * eps)
        worst = max(worst, abs(num - flat_g[i]) / max(abs(num) + abs(flat_g[i]), 1e-10))
    net.set_flat(x0)
    for _ in range(min(n_probe, X.size)):
        r, c = rng.integers(0, X.shape[0]), rng.integers(0, X.shape[1])
        Xp, Xm = X.copy(), X.copy()
        Xp[r, c] += eps
        Xm[r, c] -= eps
        num = (loss_of(net, Xp) - loss_of(net, Xm)) / (2 * eps)
        worst = max(worst, abs(num - dX[r, c]) / max(abs(num) + abs(dX[r, c]), 1e-10))
    return worst
