"""Toy sanity check: same-batch PrePO/SFPO vs a one-gradient PrePO-1.

Noisy, ill-conditioned least squares with fresh minibatches every step.
Reports excess population risk against (a) backward passes and (b) batches
consumed. This is a sanity check of the derivation, not evidence about LLM RL.

Run: python experiments/toy_least_squares.py [batch_size] [eta]
  e.g. 512 0.25 (mild curvature), 512 0.6 / 32 0.6 (sharp: eta*lambda_max near 1)
"""

import sys

import numpy as np

D_DIM, N, NOISE = 50, 50_000, 0.5
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 32
ETA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
K, ALPHA = 5, 0.5
GRAD_BUDGET = 3000
SEEDS = range(5)


def S(r, n):
    out, p = np.zeros_like(r), np.ones_like(r)
    for _ in range(n):
        out, p = out + p, p * r
    return out


def make_problem(seed):
    rng = np.random.default_rng(seed)
    lam = np.logspace(-4, 0, D_DIM)  # population curvatures, condition number 1e4
    X = rng.standard_normal((N, D_DIM)) * np.sqrt(lam)
    w_star = rng.standard_normal(D_DIM)
    y = X @ w_star + NOISE * rng.standard_normal(N)
    return rng, X, y, lam, w_star


def batch_grad(X, y, idx):
    def g(w):
        Xb = X[idx]
        return Xb.T @ (Xb @ w - y[idx]) / len(idx)

    return g


def risk(w, lam, w_star):
    return 0.5 * np.sum(lam * (w - w_star) ** 2)


def run(method, seed):
    rng, X, y, lam, w_star = make_problem(seed)
    w = np.zeros(D_DIM)
    grads, steps = 0, 0
    log = [(0, 0, risk(w, lam, w_star))]
    # PrePO-1 state (secant curvature EMA)
    g_prev = s_prev = None
    a_ys = np.zeros(D_DIM)
    b_ss = np.zeros(D_DIM)
    beta_h = 0.9

    while grads < GRAD_BUDGET:
        g = batch_grad(X, y, rng.integers(0, N, BATCH))
        if method == "SGD":
            w = w - ETA * g(w)
            grads += 1
        elif method == "SGD, lr x (1+aK)":
            w = w - ETA * (1 + ALPHA * K) * g(w)
            grads += 1
        elif method == "SFPO (K+1 grads)":
            x = w.copy()
            for _ in range(K):
                x = x - ETA * g(x)
            xt = w + ALPHA * (x - w)
            w = xt - ETA * g(xt)
            grads += K + 1
        elif method == "PrePO (3 grads)":
            z0 = g(w)
            th1 = w - ETA * z0
            z1 = g(th1)
            small = np.abs(z0) <= 1e-3 * np.sqrt(np.mean(z0**2))
            r = np.where(small, 1.0, np.clip(z1 / np.where(small, 1.0, z0), -1, 1))
            th_hat = th1 - ETA * S(r, K - 1) * z1
            xt = w + ALPHA * (th_hat - w)
            w = xt - ETA * g(xt)
            grads += 3
        elif method == "PrePO-1 (1 grad)":
            gt = g(w)
            if g_prev is not None:
                a_ys = beta_h * a_ys + (1 - beta_h) * (gt - g_prev) * s_prev
                b_ss = beta_h * b_ss + (1 - beta_h) * s_prev**2
            if steps < 2:
                r = np.zeros(D_DIM)  # warm-up: plain base step until the EMA has data
            else:
                Dhat = np.maximum(a_ys / (b_ss + 1e-30), 0.0)
                r = np.clip(1 - ETA * Dhat, 0.0, 1.0)
            Gam = ETA * (1 + ALPHA * r * S(r, K))  # Eq. (3.3), tau = 1
            s = -Gam * gt
            w = w + s
            g_prev, s_prev = gt, s
            grads += 1
        else:
            raise ValueError(method)
        steps += 1
        rk = risk(w, lam, w_star)
        if not np.isfinite(rk) or rk > 1e8:
            log.append((grads, steps, np.inf))
            break
        log.append((grads, steps, rk))
    return np.array(log)


def first_reach(log, col, target):
    hit = np.nonzero(log[:, 2] <= target)[0]
    return log[hit[0], col] if len(hit) else np.nan


methods = ["SGD", "SGD, lr x (1+aK)", "SFPO (K+1 grads)", "PrePO (3 grads)", "PrePO-1 (1 grad)"]
logs = {m: [run(m, s) for s in SEEDS] for m in methods}

# Target: SGD's median risk at the end of its budget (like "GRPO's peak" in the paper).
sgd_end = [lg[-1, 2] for lg in logs["SGD"]]
target = float(np.median(sgd_end))
print(f"batch={BATCH}  target excess risk = {target:.4g} (SGD's end-of-budget level)\n")
print(f"{'method':<20} {'backward passes to target':>26} {'batches to target':>18} {'final risk':>12}")
for m in methods:
    bp = [first_reach(lg, 0, target) for lg in logs[m]]
    bt = [first_reach(lg, 1, target) for lg in logs[m]]
    fin = [lg[-1, 2] for lg in logs[m]]
    fmt = lambda v: f"{np.nanmedian(v):.0f}" if not np.all(np.isnan(v)) else "not reached"
    print(f"{m:<20} {fmt(bp):>26} {fmt(bt):>18} {np.median(fin):>12.4g}")
