"""

Model (letter, Sec. II)
    y_{n,k} = sgn(x_k + sigma * xi_{n,k} - theta_m),   xi_{n,k} ~ N(0, 1) i.i.d.,
    N elements in M groups (group m: N_m elements, threshold theta_m),
    group averages s_{m,k}, decoder  xhat_k = w^T s_k.

Adaptation (letter, Sec. III)
    weights : RLS, eq. (8)  (regularization R_w, prior P_0 = P0_w * I, w_0)
    scale   : scalar EKF, eq. (10)  (Q_sigma, R_sigma, P0_sigma)
              with the measured innovation  eps_k = x_k - s_k^T w_{k-1}
              or the mean-response innovation x_k - w_{k-1}^T hbar(x_k, sigma_{k-1}).

"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import integrate, optimize, special

from dataclasses import dataclass

SQRT2 = np.sqrt(2.0)
SQRT2PI = np.sqrt(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Gaussian helpers
# ---------------------------------------------------------------------------
def gauss_pdf(u):
    return np.exp(-0.5 * np.asarray(u, dtype=float) ** 2) / SQRT2PI


def hbar(x, sigma, theta):
    """Smoothed response E[sgn(x + sigma*xi - theta) | x] = 2 Phi((x-theta)/sigma) - 1
    for Gaussian dither (letter, below eq. (2))."""
    return special.erf((x - theta) / (sigma * SQRT2))


def hbar_prime(x, sigma, theta):
    """Scale derivative of the smoothed response, letter eq. (3), Gaussian dither."""
    return -2.0 * (x - theta) / sigma ** 2 * gauss_pdf((x - theta) / sigma)


# ---------------------------------------------------------------------------
# Closed forms for one group (theta = 0), input x ~ N(0, sx^2), Gaussian dither
# ---------------------------------------------------------------------------
def p_single(sigma, sx=1.0):
    """p(sigma) = E[x hbar] = sqrt(2/pi) sx^2 / sqrt(sx^2 + sigma^2)."""
    return np.sqrt(2.0 / np.pi) * sx ** 2 / np.sqrt(sx ** 2 + np.asarray(sigma, float) ** 2)


def psi_single(sigma, sx=1.0):
    """psi(sigma) = E[hbar^2] = (2/pi) arcsin(sx^2 / (sx^2 + sigma^2))."""
    return (2.0 / np.pi) * np.arcsin(sx ** 2 / (sx ** 2 + np.asarray(sigma, float) ** 2))


def dp_single(sigma, sx=1.0):
    sigma = np.asarray(sigma, float)
    return -np.sqrt(2.0 / np.pi) * sx ** 2 * sigma / (sx ** 2 + sigma ** 2) ** 1.5


def dpsi_single(sigma, sx=1.0):
    sigma = np.asarray(sigma, float)
    return -(4.0 / np.pi) * sx ** 2 / ((sx ** 2 + sigma ** 2) * np.sqrt(sigma ** 2 + 2.0 * sx ** 2))


def zeta_min_single(sigma, N, sx=1.0):
    """Letter eq. (5): E[x^2] - N p^2 / (1 + (N-1) psi)."""
    p, psi = p_single(sigma, sx), psi_single(sigma, sx)
    return sx ** 2 - N * p ** 2 / (1.0 + (N - 1) * psi)


def w_opt_single(sigma, N, sx=1.0):
    """Wiener coefficient on the average s_k: w^o = N p / (1 + (N-1) psi)."""
    return N * p_single(sigma, sx) / (1.0 + (N - 1) * psi_single(sigma, sx))


def sigma_opt_single(N, sx=1.0):
    """MSE-optimal scale sigma^o of eq. (5); sigma^o = 0 for N = 1 (no noise benefit)."""
    if N == 1:
        return 0.0
    res = optimize.minimize_scalar(lambda s: zeta_min_single(s, N, sx), bounds=(1e-6, 10.0 * sx),
                                   method="bounded", options={"xatol": 1e-10})
    return float(res.x)


def sigma_meanfield_single(N, sx=1.0):
    """Mean-field equilibrium sigma^dagger of the scale update (letter, Proposition 1, eq. (12)):
    p'(sigma) [1 + (N-1) psi(sigma)] = (N/2) p(sigma) psi'(sigma)."""
    f = lambda s: dp_single(s, sx) * (1.0 + (N - 1) * psi_single(s, sx)) - 0.5 * N * p_single(s, sx) * dpsi_single(s, sx)
    return float(optimize.brentq(f, 1e-3 * sx, 50.0 * sx, xtol=1e-13))


# ---------------------------------------------------------------------------
# M groups, input x ~ N(0, sx^2), Gaussian dither: Wiener solution and mean field
# ---------------------------------------------------------------------------
def _expect(fun, sx=1.0, points=None):
    """E[fun(x)] for x ~ N(0, sx^2) by adaptive quadrature."""
    lo, hi = -12.0 * sx, 12.0 * sx
    pts = None if points is None else [p for p in points if lo < p < hi]
    val, _ = integrate.quad(lambda x: fun(x) * gauss_pdf(x / sx) / sx, lo, hi,
                            points=pts, limit=500, epsabs=1e-13, epsrel=1e-11)
    return val


def grouped_moments(sigma, thetas, Ns, sx=1.0):
    """p_m = E[x hbar_m] (closed form), R_mm = E[hbar_m^2] + E[1 - hbar_m^2]/N_m,
    R_mn = E[hbar_m hbar_n] (letter, Sec. II)."""
    th = np.asarray(thetas, float)
    Nm = np.asarray(Ns, float)
    tau = np.sqrt(sx ** 2 + sigma ** 2)
    p = 2.0 * sx ** 2 * gauss_pdf(th / tau) / tau
    M = th.size
    R = np.empty((M, M))
    for i in range(M):
        for j in range(i, M):
            R[i, j] = R[j, i] = _expect(lambda x, a=th[i], b=th[j]: hbar(x, sigma, a) * hbar(x, sigma, b),
                                        sx, list(th))
    R[np.diag_indices(M)] += (1.0 - np.diag(R)) / Nm
    return p, R


def zeta_min_grouped(sigma, thetas, Ns, sx=1.0):
    """Minimum MSE over the group weights and the Wiener weights: (zeta_min, w_opt)."""
    p, R = grouped_moments(sigma, thetas, Ns, sx)
    w = np.linalg.solve(R, p)
    return float(sx ** 2 - p @ w), w


def sigma_opt_grouped(thetas, Ns, sx=1.0, bounds=(0.02, 5.0)):
    res = optimize.minimize_scalar(lambda s: zeta_min_grouped(s, thetas, Ns, sx)[0],
                                   bounds=(bounds[0] * sx, bounds[1] * sx), method="bounded",
                                   options={"xatol": 1e-8})
    return float(res.x)


def drift_grouped(sigma, thetas, Ns, sx=1.0):
    """Mean field of the scale update, letter eq. (11), evaluated at w = w^o(sigma):
    b = E[(w^T hbar') (x - w^T hbar)]."""
    th = np.asarray(thetas, float)
    _, w = zeta_min_grouped(sigma, th, Ns, sx)

    def f(x):
        return (w @ hbar_prime(x, sigma, th)) * (x - w @ hbar(x, sigma, th))

    return _expect(f, sx, list(th))


def sigma_meanfield_grouped(thetas, Ns, sx=1.0, grid=None):
    """Equilibrium sigma^dagger of the scale update for M groups (root of the drift)."""
    grid = np.linspace(0.05, 4.0, 80) * sx if grid is None else np.asarray(grid)
    vals = [drift_grouped(s, thetas, Ns, sx) for s in grid]
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if fa > 0 >= fb:   # stable root: drift changes from upward to downward
            return float(optimize.brentq(drift_grouped, a, b, args=(thetas, Ns, sx), xtol=1e-10))
    raise RuntimeError("no stable root of the mean-field drift found on the grid")


# ---------------------------------------------------------------------------
# Configurations and the vectorized adaptive simulation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArrayConfig:
    thetas: Tuple[float, ...] = (0.0,)
    Ns: Tuple[int, ...] = (31,)

    @property
    def N(self) -> int:
        return int(sum(self.Ns))

    @property
    def M(self) -> int:
        return len(self.Ns)

    def slices(self):
        out, a = [], 0
        for n in self.Ns:
            out.append(slice(a, a + n))
            a += n
        return out


@dataclass(frozen=True)
class AdaptConfig:
    # RLS weights, letter eq. (8)
    R_w: float = 0.1
    P0_w: float = 10.0             # P_0 = P0_w * I
    w0: float = 1.0                # initial weight of every group
    # scale update
    scale_rule: str = "ekf"        # 'ekf' (eq. (10)), 'fixed', or 'grad' (gradient control G)
    Q_sigma: float = 1e-3
    R_sigma: float = 1.0
    P0_sigma: float = 2.0
    innovation: str = "measured"   # 'measured' eps_k, or 'mean' (mean-response innovation)
    jacobian_weights: str = "updated"  # 'updated' (w_k, as in the notebooks) or 'previous' (w_{k-1})
    clip: Optional[float] = 0.2    # max |sigma_k - sigma_{k-1}| (None: no clipping)
    bounds: Optional[Tuple[float, float]] = (0.03, 5.0)  # sigma_k kept in [lo, hi] (None: no bounds)
    mu: float = 0.0                # step size of the gradient control (log-sigma)
    grad_log_clip: Optional[float] = 1.0  # guard on |log-step| of the gradient control


# Settings used in the letter (Sec. V): measured innovation, H_k at the updated
# weights, safeguards as stated in the letter.
SINGLE = ArrayConfig(thetas=(0.0,), Ns=(31,))
TWO_GROUPS = ArrayConfig(thetas=(-0.5, 0.5), Ns=(15, 16))
THREE_GROUPS = ArrayConfig(thetas=(-0.5, 0.0, 0.5), Ns=(21, 21, 21))
CFG_THREE_GROUPS = AdaptConfig(w0=1.0 / 3.0, clip=0.5, bounds=(0.03, 5.0))
CFG_SINGLE = AdaptConfig(w0=1.0, clip=0.2, bounds=(0.03, 5.0))
CFG_TWO_GROUPS = AdaptConfig(w0=0.5, clip=0.5, bounds=(0.03, 5.0))
SIGMA0_LIST = (0.1, 0.5, 1.5, 4.0)
N_TRIALS = 100
N_SAMPLES = 3000
LATE_START = 1500          # late window = samples with 0-based index >= 1500
MASTER_SEED = 2026


def run_adaptive(x, xi, arr: ArrayConfig, cfg: AdaptConfig, sigma0, stream_idx=None,
                 sigma_fixed=None, mu=None, record=True):
    """Simulate the adaptive SSR array for a batch of B runs in parallel.

    x          : (S, K) input samples of S streams (a 1-D array is one stream).
    xi         : (S, K, N) standardized dither of the same streams, N = sum(Ns).
    stream_idx : (B,) stream used by each batch row (default: one row per stream).
    sigma0     : scalar or (B,) initial scale (ignored for 'fixed').
    sigma_fixed: scalar or (B,) scale for cfg.scale_rule == 'fixed'.
    mu         : scalar or (B,) step size for cfg.scale_rule == 'grad' (default cfg.mu).

    At time k the array operates with sigma_{k-1}; the prediction uses w_{k-1};
    then RLS updates w and the scale rule updates sigma (letter, Sec. III).
    Returns a dict with trajectories (if record) and the final state.
    """
    x = np.atleast_2d(np.asarray(x, float))
    xi = np.asarray(xi)                      # float32 or float64; not copied
    if xi.ndim == 2:
        xi = xi[None]
    S, K = x.shape
    if stream_idx is None:
        stream_idx = np.arange(S)
    stream_idx = np.asarray(stream_idx, int)
    B, M, N = stream_idx.size, arr.M, arr.N
    th = np.asarray(arr.thetas, float)
    Nm = np.asarray(arr.Ns, float)
    sl = arr.slices()
    if xi.shape[-1] != N:
        raise ValueError("dither has %d columns, array has %d elements" % (xi.shape[-1], N))

    w = np.full((B, M), float(cfg.w0))
    P = np.broadcast_to(np.eye(M) * cfg.P0_w, (B, M, M)).copy()
    if cfg.scale_rule == "fixed":
        sigma = np.broadcast_to(np.asarray(sigma_fixed, float), (B,)).copy()
    else:
        sigma = np.broadcast_to(np.asarray(sigma0, float), (B,)).copy()
    Psig = np.full(B, float(cfg.P0_sigma))
    mu_b = np.broadcast_to(np.asarray(cfg.mu if mu is None else mu, float), (B,))
    failed = np.zeros(B, bool)

    if record:
        sig_tr = np.empty((B, K)); w_tr = np.empty((B, K, M)); H_tr = np.zeros((B, K))
        xhat_tr = np.empty((B, K)); err2_tr = np.empty((B, K))

    for k in range(K):
        xk = x[stream_idx, k]
        xik = xi[stream_idx, k, :]
        s = np.empty((B, M))
        for m in range(M):
            s[:, m] = np.sign(xk[:, None] + sigma[:, None] * xik[:, sl[m]] - th[m]).mean(axis=1)
        xhat = np.einsum("bm,bm->b", w, s)
        eps = xk - xhat
        if record:
            sig_tr[:, k] = sigma; xhat_tr[:, k] = xhat; err2_tr[:, k] = eps ** 2

        # --- RLS weights, eq. (8)
        Ps = np.einsum("bij,bj->bi", P, s)
        den = np.einsum("bi,bi->b", s, Ps) + cfg.R_w
        g = Ps / den[:, None]
        w_prev = w
        w = w + g * eps[:, None]
        P = P - np.einsum("bi,bj->bij", g, Ps)

        # --- scale update
        if cfg.scale_rule == "fixed":
            if record:
                w_tr[:, k] = w
            continue
        ss = np.maximum(sigma, 1e-2)                       # as in the notebooks
        dx = xk[:, None] - th[None, :]
        hp = -2.0 * dx / ss[:, None] ** 2 * gauss_pdf(dx / ss[:, None])     # hbar'
        if cfg.scale_rule == "ekf":
            wj = w if cfg.jacobian_weights == "updated" else w_prev
            H = np.einsum("bm,bm->b", wj, hp)
            if cfg.innovation == "measured":
                e = eps
            else:
                e = xk - np.einsum("bm,bm->b", w_prev, special.erf(dx / (ss[:, None] * SQRT2)))
            Kg = Psig * H / (H * H * Psig + cfg.R_sigma)
            step = Kg * e
            if cfg.clip is not None:
                step = np.clip(step, -cfg.clip, cfg.clip)
            sigma_new = sigma + step
            Psig = (1.0 - Kg * H) * Psig + cfg.Q_sigma
            if record:
                H_tr[:, k] = H
        elif cfg.scale_rule == "grad":
            hb = special.erf(dx / (ss[:, None] * SQRT2))
            resid = xk - np.einsum("bm,bm->b", w_prev, hb)
            # d/dsigma of the conditional finite-array loss (letter eq. (4), M groups)
            dloss = (-2.0 * resid * np.einsum("bm,bm->b", w_prev, hp)
                     - 2.0 * np.einsum("bm,bm->b", w_prev ** 2 / Nm[None, :], hb * hp))
            logstep = -mu_b * sigma * dloss
            if cfg.grad_log_clip is not None:
                logstep = np.clip(logstep, -cfg.grad_log_clip, cfg.grad_log_clip)
            sigma_new = sigma * np.exp(logstep)
        else:
            raise ValueError(cfg.scale_rule)
        if cfg.bounds is not None:
            sigma_new = np.clip(sigma_new, cfg.bounds[0], cfg.bounds[1])
        bad = ~np.isfinite(sigma_new) | (sigma_new <= 0)
        if bad.any():                                      # unprojected recursion left its domain
            failed |= bad
            sigma_new = np.where(bad, np.where(np.isfinite(sigma), np.abs(sigma), 1.0) + 1e-6, sigma_new)
        sigma = sigma_new
        if record:
            w_tr[:, k] = w

    out = {"w": w, "sigma": sigma, "P": P, "P_sigma": Psig, "failed": failed}
    if record:
        out.update(sigma_traj=sig_tr, w_traj=w_tr, H_traj=H_tr, xhat=xhat_tr, err2=err2_tr)
    return out


def predict_frozen(x, xi, arr: ArrayConfig, w, sigma):
    """Reconstruction with frozen weights w (B, M) and scale sigma (B,).
    x: (B, K), xi: (B, K, N). Returns xhat (B, K)."""
    x = np.atleast_2d(np.asarray(x, float))
    th = np.asarray(arr.thetas, float)
    sigma = np.asarray(sigma, float).reshape(-1, 1, 1)
    xhat = np.zeros_like(x)
    for m, sl in enumerate(arr.slices()):
        sm = np.sign(x[..., None] + sigma * xi[..., sl] - th[m]).mean(axis=-1)
        xhat += np.asarray(w)[:, m][:, None] * sm
    return xhat


def run_convergence_study(arr: ArrayConfig, cfg: AdaptConfig, sigma0_list=SIGMA0_LIST,
                          n_trials=N_TRIALS, K=N_SAMPLES, seed=MASTER_SEED, tag=0):
    ss = np.random.SeedSequence([seed, tag])
    n0 = len(sigma0_list)
    
    # 1. 生成 n_trials 条输入信号 x（所有 sigma0 共享同一组输入信号 x）
    rngs_x = [np.random.default_rng(c) for c in ss.spawn(n_trials)]
    x = np.stack([r.standard_normal(K) for r in rngs_x])
    stream_idx = np.tile(np.arange(n_trials), n0)
    
    # 2. 生成 n0 * n_trials 条独立的抖动噪声 xi（每个 sigma0 拥有各自独立的噪声）
    rngs_xi = [np.random.default_rng(c) for c in ss.spawn(n0 * n_trials)]
    xi = np.stack([r.standard_normal((K, arr.N)) for r in rngs_xi])
    
    sigma0 = np.repeat(np.asarray(sigma0_list, float), n_trials)
    
    # 注意：为了让 run_adaptive 支持每个 batch 独立的 xi，将 x 按 stream_idx 展开传入
    out = run_adaptive(x[stream_idx], xi, arr, cfg, sigma0, stream_idx=np.arange(n0 * n_trials))
    
    res = {}
    for i, s0 in enumerate(sigma0_list):
        rows = slice(i * n_trials, (i + 1) * n_trials)
        res[s0] = {key: out[key][rows] for key in
                   ("sigma_traj", "w_traj", "H_traj", "xhat", "err2", "failed")}
        res[s0]["x"] = x
    return res


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def moving_average(e2, window, partial=True):
    """Causal moving average along the last axis. For the first window-1 samples the
    mean over the available samples is used (partial=True) or NaN (partial=False)."""
    e2 = np.asarray(e2, float)
    K = e2.shape[-1]
    c = np.cumsum(e2, axis=-1)
    out = np.empty_like(c)
    lag = np.concatenate([np.zeros(c.shape[:-1] + (1,)), c[..., :K - window]], axis=-1)
    out[..., window - 1:] = (c[..., window - 1:] - lag) / window
    if partial:
        out[..., :window - 1] = c[..., :window - 1] / np.arange(1, window)
    else:
        out[..., :window - 1] = np.nan
    return out


def first_passage(e2, threshold, window=50):
    """First time k >= window (1-based) at which the causal window-sample moving
    average of e2 falls below threshold (letter, Sec. V-A). NaN if never."""
    ma = moving_average(e2, window, partial=False)
    e2 = np.atleast_2d(e2)
    ma = np.atleast_2d(ma)
    T = np.full(e2.shape[0], np.nan)
    for i in range(e2.shape[0]):
        idx = np.flatnonzero(ma[i] < threshold)
        if idx.size:
            T[i] = idx[0] + 1
    return T


# ---------------------------------------------------------------------------
# Speech test signal (letter, Sec. V-D; identical to the original notebook)
# ---------------------------------------------------------------------------
def synth_vowel_a(fs=8000, duration=3.0, F0=120.0, harmonics=25,
                  formants=((800.0, 80.0), (1200.0, 100.0), (2500.0, 150.0)), peak=0.9):
    """Synthesized vowel /a/: 25 harmonics (amplitude 1/k) of F0 through three
    second-order formant resonators, normalized to a peak of 0.9."""
    from scipy.signal import lfilter
    t = np.arange(0.0, duration, 1.0 / fs)
    src = np.zeros_like(t)
    for k in range(1, harmonics + 1):
        src += np.sin(2.0 * np.pi * k * F0 * t) / k
    y = src
    for fc, bw in formants:
        r = np.exp(-np.pi * bw / fs)
        th = 2.0 * np.pi * fc / fs
        y = lfilter([1.0 - r], [1.0, -2.0 * r * np.cos(th), r * r], y)
    y = y / np.max(np.abs(y)) * peak
    return t, y


# ---------------------------------------------------------------------------
# Plot helpers (EPS-safe: shading by lightened colors instead of transparency)
# ---------------------------------------------------------------------------
def set_plot_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 15, "axes.labelsize": 15, "legend.fontsize": 13,
        "axes.unicode_minus": False,
        "ps.fonttype": 42, "pdf.fonttype": 42,
    })


def lighten(color, amount=0.3):
    """Mix a color with white; amount = 0 gives white, 1 the original color."""
    import colorsys
    import matplotlib.colors as mc
    h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(color))
    return colorsys.hls_to_rgb(h, 1.0 - amount * (1.0 - l), s)


def save_figure(fig, stem, out_dir="."):
    """Save stem.eps (used by the letter), stem.pdf and stem.png."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    for ext, kw in (("eps", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), bbox_inches="tight", **kw)
