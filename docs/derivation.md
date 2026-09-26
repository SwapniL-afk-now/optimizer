# Folding PrePO into a one-gradient optimizer: derivation notes

**Question.** SFPO and PrePO get more progress per rollout batch by running several
optimizer transitions on the same batch: K + 1 backward passes for SFPO, 3 for PrePO.
Can the probe → predict → reposition → correct process be written as a single optimizer
update, with one backward pass per batch like AdamW or Muon, that keeps most of the benefit?

**Short answer.**

1. On a quadratic batch loss, the whole three-stage process is *exactly* one
   preconditioned gradient step. The preconditioner is a polynomial in the batch Hessian:
   a truncated Neumann series, i.e. a partial Newton step (§2).
2. PrePO approximates that polynomial with a per-coordinate trend `r`. Once `r` is known,
   PrePO's three passes collapse into one step with a closed-form per-coordinate step size
   `Γ(r)` (§3). The extra backward passes only buy `r`, plus a correction gradient that the
   same model can predict.
3. `r` can be estimated from the optimizer's own history across batches, with no extra
   backward pass. This estimate is exact in the deterministic limit. With SGD motions,
   noise makes it shrink toward the base optimizer on its own, a soft version of PrePO's
   entropy gate. With momentum, an explicit SNR factor is needed, and Adam's moments
   provide one for free. The estimate measures the trend of the *population* objective
   rather than of the current batch.
   Because it is fixed before the batch is drawn, it removes the extra variance term in
   PrePO's Corollary 1 (§4).
4. The result is an AdamW/Muon variant with a trend-dependent per-coordinate step size (§5).
   A toy check supports the algebra and the trade-off (§7). It is not evidence about LLM RL.

What this does **not** give: more information per batch. A one-gradient method sees each
batch once. It recovers PrePO's *step geometry* (how far to move in each coordinate) from
history. It does not recover PrePO's *batch reuse*. Whether that is enough in LLM RL is the
main empirical question (§6, §8).

Working name below: **PrePO-1** (PrePO with one backward pass).

Status tags: **[proved]**, **[verified numerically]** (see `experiments/verify_identities.py`),
**[sketch]**, **[conjecture]**.

---

## 1. Setup and notation

Notation follows the PrePO paper. One outer step at parameters $\theta_t$ with batch $B_t$:

- batch loss $L_B$, batch gradient $g_t = \nabla L_B(\theta_t)$, batch Hessian $H_B$;
- inner learning rate $\eta$ (the virtual trajectory), correction learning rate
  $\eta_c = \tau\eta$ (the paper uses $\tau = 1$ in RL), horizon $K$, blend $\alpha$;
- a frozen diagonal preconditioner $P = \mathrm{diag}(p)$ within the step (AdamW:
  $p = 1/(\sqrt{\hat v}+\epsilon)$; SGD: $p = 1$). This is the same assumption as the paper's
  Appendix B;
- $S_n(r) = \sum_{q=0}^{n-1} r^q$, applied elementwise to diagonal $R = \mathrm{diag}(r)$.

PrePO step (Algorithm 1, SGD motions, no weight decay):

$$
\begin{aligned}
z_0 &= g(\theta_t), \quad \theta_1 = \theta_t - \eta z_0, \quad z_1 = g(\theta_1),
\quad r_i = \Pi_{[-1,1]}(z_{1,i}/z_{0,i}),\\
\hat\theta &= \theta_1 - \eta S_{K-1}(R) z_1, \quad
\tilde\theta = \theta_t + \alpha(\hat\theta - \theta_t), \quad
\theta_{t+1} = \tilde\theta - \eta_c\, g(\tilde\theta).
\end{aligned}
$$

That is three gradient evaluations on the same batch. SFPO replaces the prediction with K
real steps, which costs K + 1 evaluations.

---

## 2. The three stages are one polynomial-preconditioned step (quadratic case)

Let $L_B(\theta) = \tfrac12(\theta-\theta^\*)^\top H_B(\theta-\theta^\*)$, and let
$\theta^{(n)}$ be the point reached after $n$ GD steps of size $\eta$ from $\theta_t$.

**Lemma 1 (SFPO is an affine blend) [proved, verified numerically].** With $\eta_c = \eta$,
on any quadratic (no diagonal assumption),

$$
\theta_{t+1}^{\text{SFPO}} = (1-\alpha)\,\theta^{(1)} + \alpha\,\theta^{(K+1)} .
$$

*Proof.* One GD step $T(x) = x - \eta H_B(x-\theta^\*)$ is an affine map, and affine maps
preserve affine combinations:
$T(\tilde\theta) = T((1-\alpha)\theta_t + \alpha\theta^{(K)}) = (1-\alpha)\theta^{(1)} + \alpha\theta^{(K+1)}$. ∎

**Corollary 1 (the optimizer equation hiding inside SFPO/PrePO).** Write
$Q_n(H) = \eta\sum_{q<n}(I-\eta H)^q = H^{-1}\big(I-(I-\eta H)^n\big)$, the $n$-term Neumann
approximation of $H^{-1}$ that $n$ GD steps apply. Then

$$
\boxed{\;\theta_{t+1} = \theta_t - \Gamma_{K,\alpha}(H_B)\, g_t,\qquad
\Gamma_{K,\alpha}(H) = (1-\alpha)\,\eta I + \alpha\, Q_{K+1}(H)
= \eta\Big[I + \alpha\sum_{q=1}^{K}(I-\eta H)^q\Big].\;}
$$

For general $\eta_c$: $\Gamma = \alpha Q_K(H) + \eta_c\,(I - \alpha H Q_K(H))$. With the frozen
preconditioner, replace $H \to PH$ and $g \to Pg$.

**Reading.**

- Each eigen-direction with curvature $h$ is multiplied by
  $\varphi(h) = (1-\eta_c h)\big(1-\alpha+\alpha(1-\eta h)^K\big)$. This is exactly the
  paper's Proposition 2 filter.
- As $K\to\infty$ (with $\tau = 1$ and $0 < \eta h < 2$): $\Gamma \to (1-\alpha)\,\eta I + \alpha H_B^{-1}$. **SFPO/PrePO interpolate
  between a gradient step ($\alpha = 0$) and a Newton step on the batch loss
  ($\alpha = 1$, $K = \infty$).** $K$ sets the Neumann truncation. $\alpha$ sets how much of
  the batch-Newton step is trusted.
- This makes "don't fully commit" precise. $H_B$ and $g_B$ are one batch's estimates, so the
  full batch-Newton step overfits the batch. $\alpha < 1$ acts as damping (trust region).
- This is also what "richer signal per step" means mathematically. The extra same-batch
  passes deliver second-order information about the batch loss along the optimizer's own
  motion. Each extra pass is effectively one Hessian–vector product by finite differences:
  $g(\theta_1) - g(\theta_t) \approx -\eta H_B P z_0$.

---

## 3. Given the trend, PrePO is one step with a closed-form step size

PrePO does not form the matrix polynomial. It fits a diagonal trend $R$ with $Rz_0 \approx z_1$.
On a diagonal quadratic the fit is exact: $r_i = 1 - \eta p_i h_i$.

**Proposition 1 (one-gradient form of PrePO) [proved, verified numerically].** On a diagonal
quadratic with SGD (or fixed-$P$) motions, and with the clip inactive, the full PrePO step
(both probes, prediction, reposition, correction) equals

$$
\boxed{\;\theta_{t+1} = \theta_t - \Gamma(R)\,P g_t,\qquad
\Gamma(r) = \alpha\eta\, S_K(r) + \eta_c\,\big(1-\alpha+\alpha r^K\big),\;}
$$

and for $\tau = 1$ ($\eta_c = \eta$),

$$
\boxed{\;\Gamma(r) = \eta\Big(1 + \alpha\sum_{q=1}^{K} r^q\Big) = \eta\Big(1 + \alpha\, r\,\frac{1-r^K}{1-r}\Big).\;}
$$

*Proof.* $z_1 = r z_0$, so $\hat\theta = \theta_t - \eta(1 + rS_{K-1}(r))z_0 = \theta_t - \eta S_K(r) z_0$
and $\tilde\theta = \theta_t - \alpha\eta S_K(r) z_0$. The correction gradient is
$g(\tilde\theta) = z_0 - h\alpha\eta S_K(r) z_0 = (1-\alpha(1-r^K))\,z_0$, using
$\eta h S_K(r) = 1-r^K$. Adding the two moves gives $\Gamma(r)$. The $\tau=1$ form follows from
$S_K(r) - 1 + r^K = \sum_{q=1}^K r^q$. ∎

**Properties.**

| trend $r_i$ | meaning | step $\Gamma(r_i)$ ($\tau = 1$) | $K{=}5,\ \alpha{=}0.5$ |
|---|---|---|---|
| $1$ | motion keeps its pace (flat) | $(1+\alpha K)\eta$, the paper's $\gamma_{hi}$ | $3.5\eta$ |
| $0$ | motion stops after one step | $\eta$, **exactly the base optimizer** | $1.0\eta$ |
| $-1$ ($K$ odd) | motion reverses (overshoot) | $(1-\alpha)\eta$ | $0.5\eta$ |

- $1 - h\,\Gamma(1-\eta h) = \varphi(h)$, so the one-gradient step reproduces PrePO's
  per-batch contraction exactly, including its stability up to $h = 2/\eta$.
- Reading: take the base step, then add $\alpha$ times the discounted sum of the $K$
  predicted future motions, with discount $r$. The trend acts as a per-coordinate discount
  factor over a horizon of $K$.
- **Consequence:** once $R$ is known, the three backward passes are one elementwise-scaled
  step. The second probe exists only to measure $R$. The correction gradient is predicted
  by the same linear model.

**PrePO's trend with momentum (AdamW) [derived].** With first-moment coefficient $\beta$ and
a diagonal $H_B$, the two probes give

$$
r_i = \underbrace{\beta}_{\text{inertia}} + (1-\beta)\underbrace{\frac{g_{t,i}}{\bar m_{t,i}}}_{\text{consistency (free)}} - \underbrace{(1-\beta)\,\eta\, p_i h_i}_{\text{curvature (needs probe 2)}},
\qquad \bar m_t = \beta m_{t-1} + (1-\beta) g_t .
$$

Only the curvature term needs the second backward pass. The other two terms are already
available from the base optimizer.

---

## 4. Estimating the trend without a second backward pass

A one-gradient method can only learn about curvature from *earlier* batches. Two
equivalent estimators:

- **Secant (recommended).** With $s_t = \theta_t - \theta_{t-1}$ and $y_t = g_t - g_{t-1}$
  (consecutive batches):
  $$\hat h_{t,i} = \frac{\mathrm{EMA}[y_{t,i}\,s_{t,i}]}{\mathrm{EMA}[s_{t,i}^2]},\qquad
  r_{t,i} = \Pi_{[0,1]}\big(1 - (1-\beta)\,\eta\, p_{t,i}\,\hat h_{t,i}\big).$$
- **Lag-1 motion trend.** $\hat\rho_i = \mathrm{EMA}[u_{t,i}u_{t-1,i}]/\mathrm{EMA}[u_{t-1,i}^2]$
  is the trend under the *actual* step $\Gamma$, not under $\eta$. Convert with
  $\hat h_i\Gamma_i \approx 1-\hat\rho_i$ (for $\beta = 0$), then proceed as above.

### 4.1 Deterministic limit: exact

For full-batch GD on a diagonal quadratic, $y_t = H s_t$, so $\hat h = h$ exactly. PrePO-1
then takes the same per-step multiplier $\varphi(h)$ that PrePO takes per batch, with one
backward pass instead of three. **[proved]**

### 4.2 Stochastic: the estimate gates itself

**Proposition 2 (SNR gating) [derived for a 1-D noisy quadratic, verified numerically].**
Take $f = \tfrac12 h x^2$, gradients $hx_t + \xi_t$ with i.i.d. noise of variance
$\sigma^2$, and SGD motions of size $\eta$. Split each motion into a deterministic part
(power $S$) and a stationary noise part (power $N = 2\eta^2\sigma^2/(2-\eta h)$), and let
$w = S/(S+N)$. Then

$$
\mathbb{E}\text{-trend} = w\,(1-\eta h) + (1-w)\left(-\frac{\eta h}{2}\right),\qquad
\hat h_{\text{secant}} = w\,h + (1-w)\left(\frac h2 + \frac1\Gamma\right).
$$

- **Drift phase** ($w\to1$): the estimate is PrePO's trend $1-\eta h$, so PrePO-1 extrapolates.
- **Stationary phase** ($w\to 0$): consecutive motions are *anti*-correlated with trend
  $-\eta h/2$. This is Pflug's classic stationarity signal. With the secant estimate the
  self-consistent trend $r = 1 - \eta/\Gamma(r)$ contracts to $r^\* = 0$ at rate $\alpha$.
  **PrePO-1 falls back to the base optimizer by itself.**
- This is a soft, per-coordinate version of PrePO's entropy gate. The paper found that
  prediction helps while motion is regular and hurts later (Appendix C.6). A cross-step
  trend switches the prediction off exactly when the motion stops being predictable.
  Whether this replaces the gate in LLM RL is a **[conjecture]**, because the paper's
  no-fallback ablation drops from 38.9 to 31.1.

**With momentum the fallback is not automatic [derived].** For $\beta > 0$ the trend
contains the inertia term $\beta$ (§3). In the noise phase the population consistency term
goes to about 0 and the curvature term shrinks by $(1-\beta)$. The secant noise inflation
works out to $\kappa \approx (1-\beta^2)\,\eta/\Gamma$. So $r$ stays near $\beta$, and the
step stays near $(1+\alpha\sum_q\beta^q)\,\eta$, about $2.8\eta$ for $\beta = 0.9$, $K = 5$,
$\alpha = 0.5$. The same happens in PrePO itself: its Muon trend sits at about 0.96, close
to the momentum coefficient. This is why PrePO with AdamW/Muon needs an explicit gate.

For momentum bases, add the SNR factor explicitly. It is free from Adam's moments. With
slowly varying signal $\mu$, $\mathbb{E}[m^2]/\mathbb{E}[v] = w' + c_\beta(1-w')$, where
$c_\beta = (1-\beta_1)/(1+\beta_1)$ and $w' = \mu^2/(\mu^2+\sigma^2)$. So

$$
w_i = \Pi_{[0,1]}\!\left(\frac{\hat m_i^2/\hat v_i - c_\beta}{1-c_\beta}\right),\qquad
r_i \leftarrow w_i\, r_i ,
$$

which rebuilds the structure of Proposition 2 ($r = w\cdot$drift trend $+\,(1-w)\cdot 0$).
Use the moments from step $t-1$ to keep the trend predictable (§4.4). **[conjecture: needs
testing]**

### 4.3 Population trend instead of batch trend

With momentum and $P$ held fixed, the cross-step ratio and PrePO's same-batch ratio have
the same form:

$$
\frac{u_{t,i}}{u_{t-1,i}} = \beta + (1-\beta)\frac{g_i(\theta_t; B_t)}{m_{t-1,i}}
\qquad\text{vs.}\qquad
\frac{z_{1,i}}{z_{0,i}} = \beta + (1-\beta)\frac{g_i(\theta_1; B_t)}{\bar m_i}.
$$

In effect, **the second probe is the next real step, taken on a fresh batch.** Conditioned
on the past, $\mathbb{E}[g(\theta_t; B_t)] = \nabla f(\theta_t)$. So the cross-step trend
estimates the trend of the *population* objective, whereas PrePO's trend is that of the
current batch loss. This is a formal version of the "multiple steps overfit the batch"
intuition. The cost is variance, which the EMA controls.

### 4.4 A predictable trend removes PrePO's extra variance term

PrePO's Corollary 1 pays $28\varrho(1-\beta)\kappa_P\sigma^2$ because the trend is fitted
on the same batch as the gradient. The paper notes that this term "would vanish for a trend
fixed in advance, but such a trend could not see which coordinates overshoot." A
history-based trend is fixed in advance ($\mathcal F_{t-1}$-measurable), and it does see
overshoot, one step late.

**Proposition 3 (rate is preserved; SGD base) [sketch].** Let $f$ be $L$-smooth with
$f_{\inf} > -\infty$. Let $\mathbb{E}[g_t\mid\mathcal F_{t-1}] = \nabla f(\theta_t) + b_t$
with $\|b_t\|\le\bar b$ and variance at most $\sigma^2$. Let
$\Gamma_t = \mathrm{diag}(\Gamma(r_t))$ be $\mathcal F_{t-1}$-measurable with
$\eta_{lo} I \preceq \Gamma_t \preceq \eta_{hi} I$, and $L\eta_{hi}^2 \le \eta_{lo}/4$. Then

$$
\frac1T\sum_{t<T}\mathbb{E}\|\nabla f(\theta_t)\|^2 \le
\frac{2(f(\theta_0)-f_{\inf})}{\eta_{lo}T} + \frac{L\eta_{hi}^2\sigma^2}{\eta_{lo}}
+ 2\Big(\frac{\eta_{hi}^2}{\eta_{lo}^2} + \frac{L\eta_{hi}^2}{\eta_{lo}}\Big)\bar b^2 .
$$

With $r\in[0,1]$ and $\tau = 1$, $\eta_{hi}/\eta_{lo} = 1+\alpha K$ is a constant, so
$\eta\propto T^{-1/2}$ gives $O(T^{-1/2})$ plus a bias floor. There is no same-batch variance
term. (Proof: descent lemma, conditional unbiasedness given $\Gamma_t$, and Young's
inequality on the bias cross-term.) A *speed-up* statement in the style of the paper's
Proposition 3 still needs the drift-phase regime from §4.2. That is open work.

---

## 5. The optimizer

### 5.1 SGD / fixed-preconditioner base

$$
\theta_{t+1} = \theta_t - \Gamma(r_t)\odot p_t\odot g_t,\qquad r_t\ \text{from §4}.
$$

### 5.2 AdamW base

Run PrePO's Algorithm 1 on the surrogate gradient model
$\hat g(\theta) = \bar m_t + \hat D_t(\theta-\theta_t)$, with $\hat D_t = \mathrm{diag}(\hat h_t)$
from §4. Use the momentum $\bar m_t$ as the zeroth-order term: virtual steps on a
population model, not on this batch. Everything is diagonal, and the procedure collapses
**[derived]** to

$$
\boxed{\;
\begin{aligned}
&m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\quad
d_t = \hat m_t/(\sqrt{\hat v_t}+\epsilon),\\
&r_t = w_t\odot\Pi_{[0,1]}\!\big(1 - (1-\beta_1)\,\eta\,\hat h_t/(\sqrt{\hat v_t}+\epsilon)\big)\quad(w_t\text{: SNR factor, §4.2}),\\
&\theta_{t+1} = \theta_t - \Gamma(r_t)\odot d_t - \eta\lambda\,\theta_t,\qquad
\Gamma(r) = \alpha\eta S_K(r) + \eta_c(1-\alpha+\alpha r^K).
\end{aligned}\;}
$$

In words: **AdamW with a per-coordinate learning-rate multiplier $\Gamma(r_t)/\eta \in [1,\,1+\alpha K]$**
(for $\tau = 1$), where the multiplier comes from the virtual K-step trajectory. The
correction gradient is the same whether its zeroth-order term is $g_t$ or $\bar m_t$,
because $\beta_1 m_{t-1} + (1-\beta_1)g_t = \bar m_t$. The faithful PrePO state after
correction is $m_t\odot(1-\alpha+\alpha r_t^K)$. Storing that instead of $m_t$ is an ablation.

PrePO-faithful variant (probes use $g_t$): $r_i = \beta_1 + (1-\beta_1)g_i/\bar m_i - \kappa_i$
and $\Gamma_i = \alpha\eta S_K(r_i) + \eta_c(1-\alpha\kappa_i S_K(r_i))$, with
$\kappa_i = (1-\beta_1)\eta p_i\hat h_i$. Here the step depends on $g_t$, so the predictable
property of §4.4 is lost.

### 5.3 Muon base

Muon moves by normalized matrices, so use one scalar trend per matrix, as PrePO does:
$\rho_t = \Pi_{[0,1]}\big(\mathrm{EMA}\langle O_t, O_{t-1}\rangle_F/\mathrm{EMA}\|O_{t-1}\|_F^2\big)$
with $O_t = \mathrm{polar}(M_t)$. Because the correction direction is renormalized, it does
not shrink, and $\Gamma_{\text{Muon}}(\rho) = \alpha\eta S_K(\rho) + \eta_c \in [\gamma_{lo}, \gamma_{hi}]$.
These are the same constants as the paper's Theorem 4. Note that this version never drops
below $(\alpha+\tau)\eta$. To allow a full fallback to Muon, set $\eta_c = (1-\alpha)\eta$.

### 5.4 Cost and memory

- One backward pass per batch, plus about $K$ elementwise operations per parameter. That is
  negligible next to a backward pass.
- Per-coordinate secant state: $g_{t-1}$, $s_{t-1}$, and two EMAs. That is four extra buffers
  on top of AdamW's two, which is heavy at LLM scale. Leaner options:
  1. Per-tensor or per-row scalar curvature. Store only $s_{t-1}$ and accumulate the inner
     products $\langle g_t, s_{t-1}\rangle$ and $\langle g_{t-1}, s_{t-1}\rangle$ as scalars.
     This is one extra buffer.
  2. An Adafactor-style factored EMA.
  3. The lag-1 motion trend, which needs $u_{t-1}$ plus one EMA.

  Coarser trends also help the SNR problem in §8.

---

## 6. When PrePO-1 pays off (cost model)

Per outer step, let $R$ be rollout cost and $B$ one backward-pass unit.

| method | cost per step |
|---|---|
| GRPO / AdamW | $R + B$ |
| PrePO | $R + 3B$ |
| PrePO-1 | $R + B$ |

PrePO-1 beats PrePO in wall-clock time iff
$N_{\text{PrePO-1}}/N_{\text{PrePO}} < (R+3B)/(R+B)$, where $N$ counts steps to target.

- **RL.** The paper reports that a PrePO step costs about 1.5× a GRPO step, which gives
  $R\approx 3B$ and a threshold of **1.5**. PrePO-1 must keep at least **2/3** of PrePO's
  step reduction.
- **SFT** ($R\approx 0$): the threshold is **3**, so PrePO-1 must keep at least **1/3**. This
  is also where PrePO's 4× step saving shrank to 1.2–1.4× in time. SFT is the setting where
  PrePO-1 has the most to gain.
- PrePO-1 beats the base optimizer whenever it saves any steps.

---

## 7. Toy check

Script: `experiments/toy_least_squares.py`. Least squares with $d = 50$, feature variances
spanning $10^{-4}$ to $1$, label noise 0.5, fresh minibatches, $K = 5$, $\alpha = 0.5$,
$\tau = 1$, a budget of 3000 backward passes, and medians over 5 seeds. The target is SGD's
excess risk at the end of its budget, analogous to "GRPO's peak".

Each cell shows backward passes / batches to reach the target, then the final excess risk.

| setting | SGD | SGD, lr × 3.5 | SFPO (6 passes) | PrePO (3 passes) | **PrePO-1 (1 pass)** |
|---|---|---|---|---|---|
| mild, $\eta = 0.25$, $b = 512$ | 2482 / 2482, 2.7e-3 | 902 / 902, 1.2e-3 | 2184 / **364**, 5.0e-3 | 2697 / 899, 2.6e-3 | **1024** / 1024, **9.6e-4** |
| mild, $\eta = 0.25$, $b = 32$ | 2070 / 2070, 4.5e-3 | 2216 / 2216, 1.4e-2 | not reached, 1.2e-2 | 2392 / **798**, 1.1e-2 | **1391** / 1391, **4.2e-3** |
| sharp, $\eta = 0.6$, $b = 512$ | 2022 / 2022, 1.2e-3 | diverges | 2754 / **459**, 3.1e-3 | 2097 / 699, 1.7e-3 | **977** / 977, **8.6e-4** |
| sharp, $\eta = 0.6$, $b = 32$ | 715 / 715, 7.7e-3 | diverges | 2394 / **399**, 1.4e-2 | 1458 / 486, 2.0e-2 | **486** / 486, 8.1e-3 |

What this shows:

- **The same-batch methods use the fewest batches. PrePO-1 uses the fewest backward passes.**
  PrePO-1 recovers the step geometry, not the batch reuse. At $b = 512$ it needs 1.1–1.4×
  as many batches as PrePO (inside the 1.5 RL threshold). At $b = 32$ it needs 1.0–1.7×,
  which is weaker, as §4.2 predicts for low SNR.
- In the sharp setting a uniformly larger learning rate diverges, while PrePO-1 stays
  stable. This matches Proposition 2 of the paper, which PrePO-1 inherits. In the mild
  setting the larger learning rate is competitive: the trend only matters when sharp
  coordinates exist.
- In the noisy settings the same-batch methods end at a higher noise floor. PrePO-1 ends
  near SGD's floor, consistent with the fallback in §4.2.

A toy linear problem says nothing about LLM RL. It checks the algebra and shows the
trade-off.

---

## 8. Risks and open questions

1. **Low per-coordinate SNR in LLMs.** $\hat r$ may collapse to 0 for most coordinates,
   making PrePO-1 ≈ AdamW. Mitigations: per-row, per-column or per-tensor trends, and longer
   EMAs.
2. **Feedback loop.** $\Gamma_t$ depends on $r_t$, which depends on past $\Gamma$. EMA lag
   could cause oscillation. The clip range and EMA timescale need analysis.
3. **Gate.** The endogenous fallback is a conjecture at LLM scale. Keep the entropy gate as
   a safety switch in the first experiments.
4. **Frozen preconditioner.** Like the paper, the derivation freezes $P$ within a step. In
   PrePO-1 it changes every step.
5. **Novelty positioning.** These were checked from memory and should be verified.
   - Adan (Xie et al., 2022): Nesterov momentum without the extrapolated-point gradient,
     via gradient differences. This is the closest precedent for turning a multi-point
     method into a one-gradient one.
   - Neumann optimizer (Krishnan, Xiao & Saurous, ICLR 2018): an amortized Neumann series
     across minibatches.
   - Lookahead (Zhang et al., 2019).
   - Hypergradient descent (Baydin et al., ICLR 2018): learning rate adapted by
     $\langle g_t, g_{t-1}\rangle$.
   - Delta-bar-delta (Jacobs, 1988) and Rprop (Riedmiller & Braun, 1993).
   - Cautious optimizers (Liang et al., 2024).
   - Apollo, a diagonal secant quasi-Newton method (Ma, 2020).
   - Pflug's test and SASA (Lang, Xiao & Zhang, NeurIPS 2019).
   - QHM (Ma & Yarats, 2019).

   The contribution to claim is the *derivation*: a step-size map $\Gamma(r)$ that reproduces
   the SFPO/PrePO spectral filter exactly, plus the batch-trend vs population-trend
   argument. The idea of scaling a learning rate by correlation is not new on its own.

## 9. Next steps

1. **Decisive diagnostic (cheap, uses existing PrePO code).** On the same RL batches, log
   PrePO's same-batch trend $r$ and the cross-step estimate $\hat r$, and compare their
   distributions and correlation over training. If they agree during the prediction phase,
   PrePO-1 should keep most of the gain.
2. Implement PrePO-1 as a drop-in `torch.optim` for AdamW and Muon, with per-tensor and
   per-coordinate trend options.
3. Run SFT on knights-and-knaves first, where the threshold is loosest. Then GRPO at 1.5B.
   Ablations:
   - uniform learning rate × $(1+\alpha K)$;
   - $R = I$ (weight prediction);
   - cautious AdamW and hypergradient descent;
   - trend granularity;
   - predictable vs same-step $r$;
   - stored-momentum choice;
   - with and without the entropy gate.
4. Theory: a speed-up bound (the paper's Proposition 3 analogue) in the drift phase, and a
   momentum/Adam version of Proposition 3 here with a predictable trend.
