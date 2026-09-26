"""Numerical checks for the identities in docs/derivation.md.

Run: python experiments/verify_identities.py
Every check prints the max deviation; identities should hold to ~1e-12.
"""

import numpy as np

rng = np.random.default_rng(0)


def S(r, n):
    """S_n(r) = sum_{q<n} r^q, elementwise."""
    r = np.asarray(r, dtype=float)
    out, p = np.zeros_like(r), np.ones_like(r)
    for _ in range(n):
        out, p = out + p, p * r
    return out


def gamma(r, eta, eta_c, alpha, K):
    """Per-coordinate step of the one-gradient form (Eq. 3.2 in the doc)."""
    return alpha * eta * S(r, K) + eta_c * (1 - alpha + alpha * r**K)


# ---------------------------------------------------------------------------
# Identity 1: SFPO with eta_c = eta on ANY quadratic is the affine blend
# (1 - alpha) * [1 GD step] + alpha * [K+1 GD steps].
# ---------------------------------------------------------------------------
d, K, alpha, eta = 20, 5, 0.5, 0.05
Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
H = Q @ np.diag(rng.uniform(0.1, 30.0, d)) @ Q.T  # non-diagonal SPD
xs = rng.standard_normal(d)
grad = lambda x: H @ (x - xs)


def gd(x, n, lr):
    for _ in range(n):
        x = x - lr * grad(x)
    return x


x0 = rng.standard_normal(d)
xK = gd(x0, K, eta)
x_tilde = x0 + alpha * (xK - x0)
sfpo = x_tilde - eta * grad(x_tilde)
blend = (1 - alpha) * gd(x0, 1, eta) + alpha * gd(x0, K + 1, eta)
print(f"[1] SFPO == (1-a)*x_1 + a*x_(K+1)            max|diff| = {np.abs(sfpo - blend).max():.2e}")

# Same thing as a matrix preconditioner: Gamma(H) = (1-a) eta I + a H^-1 (I - (I - eta H)^(K+1))
I = np.eye(d)
Gam = (1 - alpha) * eta * I + alpha * np.linalg.solve(H, I - np.linalg.matrix_power(I - eta * H, K + 1))
print(f"[1b] SFPO == x0 - Gamma_K,a(H) g0             max|diff| = {np.abs(sfpo - (x0 - Gam @ grad(x0))).max():.2e}")

# ---------------------------------------------------------------------------
# Identity 2: PrePO (3 gradient evaluations, SGD motions) on a diagonal
# quadratic equals ONE step x0 - Gamma(r) * g0 with r = 1 - eta*h,
# for general eta_c. It also equals SFPO there (prediction is exact).
# ---------------------------------------------------------------------------
h = rng.uniform(0.0, 1.9 / eta, d)  # keeps 1 - eta*h inside [-1, 1]
xs = rng.standard_normal(d)
gdiag = lambda x: h * (x - xs)
eta_c = 0.7 * eta
x0 = rng.standard_normal(d)

z0 = gdiag(x0)
th1 = x0 - eta * z0
z1 = gdiag(th1)
r = np.clip(z1 / z0, -1, 1)
th_hat = th1 - eta * S(r, K - 1) * z1
xt = x0 + alpha * (th_hat - x0)
prepo = xt - eta_c * gdiag(xt)

one_grad = x0 - gamma(1 - eta * h, eta, eta_c, alpha, K) * gdiag(x0)
print(f"[2] PrePO(3 grads) == x0 - Gamma(r) g0 (1 grad) max|diff| = {np.abs(prepo - one_grad).max():.2e}")

x = x0.copy()
for _ in range(K):
    x = x - eta * gdiag(x)
xt_s = x0 + alpha * (x - x0)
sfpo_d = xt_s - eta_c * gdiag(xt_s)
print(f"[2b] PrePO == SFPO on a diagonal quadratic      max|diff| = {np.abs(prepo - sfpo_d).max():.2e}")

# tau = 1 closed form: Gamma(r) = eta * (1 + alpha * sum_{q=1..K} r^q)
rr = rng.uniform(-1, 1, 1000)
lhs = gamma(rr, eta, eta, alpha, K)
rhs = eta * (1 + alpha * rr * S(rr, K))
print(f"[3] tau=1: Gamma(r) == eta(1 + a sum r^q)        max|diff| = {np.abs(lhs - rhs).max():.2e}")

# per-coordinate multiplier equals Proposition 2's phi(h)
hh = rng.uniform(0, 2 / eta, 1000)
phi_prepo = (1 - eta_c * hh) * (1 - alpha + alpha * (1 - eta * hh) ** K)
phi_one = 1 - hh * gamma(1 - eta * hh, eta, eta_c, alpha, K)
print(f"[4] 1 - h*Gamma(1-eta h) == phi(h) (Prop. 2)     max|diff| = {np.abs(phi_prepo - phi_one).max():.2e}")

# ---------------------------------------------------------------------------
# Statistics 5: cross-step trend of SGD motions on a noisy 1-D quadratic.
#   stationary lag-1 autocorrelation of u_t = x_{t+1} - x_t  ->  -eta*h/2
#   secant curvature  sum(y s)/sum(s^2)                        ->  h/2 + 1/eta
#   signal + noise mixture (ensemble at fixed t)               ->  w(1-eta h) + (1-w)(-eta h/2)
# ---------------------------------------------------------------------------
h1, eta1, sigma = 2.0, 0.1, 1.0
a = 1 - eta1 * h1
T = 2_000_000
xi = rng.standard_normal(T) * sigma
x = np.empty(T + 1)
x[0] = rng.standard_normal() * eta1 * sigma / np.sqrt(1 - a * a)
for t in range(T):
    x[t + 1] = a * x[t] - eta1 * xi[t]
g = h1 * x[:-1] + xi
u = np.diff(x)
rho = (u[1:] @ u[:-1]) / (u[:-1] @ u[:-1])
print(f"[5a] stationary lag-1 trend  {rho:+.4f}   theory -eta*h/2 = {-eta1 * h1 / 2:+.4f}")
s_, y_ = u[:-1], np.diff(g)  # s_t = x_t - x_{t-1}, y_t = g_t - g_{t-1}
h_sec = (y_ @ s_) / (s_ @ s_)
print(f"[5b] secant curvature        {h_sec:.4f}   theory h/2 + 1/eta = {h1 / 2 + 1 / eta1:.4f}")

# ensemble: chains = deterministic offset + stationary noise, measured after t0 steps
M, t0, xbar0 = 400_000, 3, 1.5
eps = rng.standard_normal(M) * eta1 * sigma / np.sqrt(1 - a * a)
xc = xbar0 + eps
us = []
for t in range(t0 + 2):
    xn = a * xc - eta1 * rng.standard_normal(M) * sigma
    us.append(xn - xc)
    xc = xn
u_prev, u_now = us[-2], us[-1]
emp = (u_now @ u_prev) / (u_prev @ u_prev)
ubar_prev = -eta1 * h1 * xbar0 * a ** (t0)  # deterministic motion at the earlier step
Sig = ubar_prev**2
Noi = 2 * eta1**2 * sigma**2 / (2 - eta1 * h1)
w = Sig / (Sig + Noi)
theory = w * (1 - eta1 * h1) + (1 - w) * (-eta1 * h1 / 2)
print(f"[5c] mixed-regime trend      {emp:+.4f}   theory w(1-eta h)+(1-w)(-eta h/2) = {theory:+.4f}  (w={w:.3f})")

# ---------------------------------------------------------------------------
# Fixed point 6: with the secant estimate at stationarity (h -> 0), the
# self-consistent virtual trend r = 1 - eta/Gamma(r) contracts to r = 0,
# i.e. the one-gradient optimizer falls back to the base step by itself.
# ---------------------------------------------------------------------------
for r0 in (0.99, 0.5, 0.1):
    rv = r0
    for _ in range(200):
        rv = np.clip(1 - eta / gamma(rv, eta, eta, alpha, K), 0, 1)
    print(f"[6] r iteration from {r0:.2f} -> {rv:.2e}")

# ---------------------------------------------------------------------------
# Statistics 7: with EMA momentum (beta > 0) at stationarity the secant
# curvature is inflated by (1+beta)/Gamma, so the momentum-scaled curvature
# kappa = (1-beta)*eta*h_sec ~ (1-beta^2)*eta/Gamma stays small: the trend
# stays near 1 and there is NO automatic fallback. Adam's m^2/v -> c_beta.
# ---------------------------------------------------------------------------
beta, hq, lr, T = 0.9, 0.01, 0.1, 2_000_000
xi = rng.standard_normal(T)
th, m, v = 0.0, 0.0, 1.0
gs, ths, ms, vs = np.empty(T), np.empty(T), np.empty(T), np.empty(T)
for t in range(T):
    gt = hq * th + xi[t]
    m = beta * m + (1 - beta) * gt
    v = 0.999 * v + 0.001 * gt * gt
    gs[t], ths[t], ms[t], vs[t] = gt, th, m, v
    th = th - lr * m
burn = 100_000
s_m = np.diff(ths)[burn:]
y_m = np.diff(gs)[burn:]
h_sec_m = (y_m @ s_m) / (s_m @ s_m)
kappa = (1 - beta) * lr * h_sec_m
print(f"[7a] momentum secant kappa   {kappa:.4f}   theory (1-b^2)*eta/Gamma + (1-b)*eta*h = {(1 - beta**2) + (1 - beta) * lr * hq:.4f}")
c_beta = (1 - beta) / (1 + beta)
print(f"[7b] stationary E[m^2]/E[v]  {np.mean(ms[burn:]**2) / np.mean(vs[burn:]):.4f}   theory c_beta = {c_beta:.4f}")
