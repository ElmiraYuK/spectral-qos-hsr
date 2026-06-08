#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectral QoS controller on a high-speed-rail corridor -- reproducible experiment.

Companion code for the paper on a periodic M/M(t)/1 model of an on-board traffic
aggregator in a 5G/6G high-speed-rail network. A trans-Alpine corridor (Paris-Milan)
is used only to position the coverage map; per the paper's data statement the corridor
geometry does NOT enter the service-rate generation. The script:

  * builds a geo-anchored service-rate profile mu(t) with five coverage gaps,
  * validates the harmonic-balance spectral solver against Monte Carlo,
  * compares six admission controllers (DropTail, Static, Reactive, Radio-map,
    the proposed Spectral+map, and a lightweight tabular Q-learning),
  * studies robustness across train speed and the proactive pre-draining mechanism.

Author : El'mira Yu. Kalimulina
         Kharkevich Institute for Information Transmission Problems, RAS;
         Lomonosov Moscow State University.

Requirements: numpy, scipy, matplotlib, pyproj, contextily, folium  (see requirements.txt).

Run (headless): python vsm_spectral_qos_experiment.py
Figures are written as PNG files in the working directory; the interactive map is
saved to coverage_map.html. For inline/interactive figures use the companion notebook.
"""

import matplotlib
matplotlib.use("Agg")          # headless: write figures to PNG instead of showing them
import warnings
warnings.filterwarnings("ignore")

# --- Section 0. Dependencies: installed via requirements.txt (numpy scipy matplotlib pyproj contextily folium) ---

# ==============================================================================
# Section 2. Numerical kernel (NumPy/SciPy)
# ==============================================================================
# -*- coding: utf-8 -*-
"""
hsr_core.py — physical/numerical core for the map-grounded periodic M/M(t)/1
edge-controller experiment.

Frame: everything physical is computed in a planar metric frame (metres).
The Colab notebook adds the geographic layer (lat/lon <-> EPSG:3857) on top.

Sections
  1. Route geometry (densified polyline, arc length)
  2. Base-station layout (irregular spacing + perpendicular offset + gaps)
  3. Propagation: path loss + spatially-correlated shadowing -> SNR(best server)
  4. SNR -> service rate mu; sample mu_full(t) along the path at constant speed
  5. Periodic component mu_per(t) by phase-folding (quasi-periodic -> periodic)
  6. Harmonic-balance spectral solver for the periodic M/M(t)/1 regime
  7. Event-driven Monte Carlo (thinning) with 5 controllers + goodput
All numerics are NumPy/SciPy only (no map dependencies here).
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# ----------------------------------------------------------------------
# 1. Route geometry
# ----------------------------------------------------------------------
def densify_polyline(waypoints_xy, step=50.0):
    """waypoints_xy: (P,2) metres. Returns densified path (M,2) and arc length s (M,)."""
    wp = np.asarray(waypoints_xy, float)
    pts = [wp[0]]
    s = [0.0]
    for a, b in zip(wp[:-1], wp[1:]):
        seg = b - a
        L = np.hypot(*seg)
        nseg = max(1, int(np.ceil(L / step)))
        for j in range(1, nseg + 1):
            p = a + seg * (j / nseg)
            pts.append(p)
            s.append(s[-1] + L / nseg)
    return np.array(pts), np.array(s)


def point_at_arclength(path_xy, s_path, s_query):
    """Linear interpolation of the path position at given arc length(s)."""
    x = np.interp(s_query, s_path, path_xy[:, 0])
    y = np.interp(s_query, s_path, path_xy[:, 1])
    return np.column_stack([x, y]) if np.ndim(s_query) else np.array([x, y])


def unit_normals(path_xy, s_path, s_query):
    """Unit left-normal of the path at given arc lengths (for BS perpendicular offset)."""
    eps = 1.0
    p1 = point_at_arclength(path_xy, s_path, np.atleast_1d(s_query) - eps)
    p2 = point_at_arclength(path_xy, s_path, np.atleast_1d(s_query) + eps)
    t = p2 - p1
    t /= (np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-12)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    return n


# ----------------------------------------------------------------------
# 2. Base-station layout (irregular -> quasi-periodic coverage)
# ----------------------------------------------------------------------
def make_bs_layout(path_xy, s_path, spacing=3000.0, jitter=0.18,
                   offset_mean=350.0, offset_jitter=300.0,
                   drop_prob=0.18, rng=None):
    """
    Place BSs along the route at ~`spacing` arc-length, with multiplicative
    jitter on the spacing, a perpendicular offset from the track, and random
    omissions (`drop_prob`) that create genuine coverage gaps.
    Returns array of BS positions (Nbs,2) in metres and their arc-length s_bs.
    """
    rng = rng or np.random.default_rng(0)
    total = s_path[-1]
    s_bs = []
    s = spacing * 0.5
    while s < total:
        s_bs.append(s)
        s += spacing * (1.0 + jitter * (2 * rng.random() - 1))
    s_bs = np.array(s_bs)
    keep = rng.random(s_bs.size) > drop_prob          # omit some -> gaps
    s_bs = s_bs[keep]
    base = point_at_arclength(path_xy, s_path, s_bs)
    nrm = unit_normals(path_xy, s_path, s_bs)
    side = rng.choice([-1.0, 1.0], size=s_bs.size)
    off = side * (offset_mean + offset_jitter * (2 * rng.random(s_bs.size) - 1))
    bs = base + nrm * off[:, None]
    return bs, s_bs


# ----------------------------------------------------------------------
# 3. Propagation: path loss + correlated shadowing -> SNR (best server)
# ----------------------------------------------------------------------
def _shadow_field(points, centers, amps, scales):
    """Smooth spatially-correlated shadowing (sum of Gaussian bumps), dB."""
    pts = np.atleast_2d(points)
    out = np.zeros(pts.shape[0])
    for c, a, sc in zip(centers, amps, scales):
        d2 = np.sum((pts - c) ** 2, axis=1)
        out += a * np.exp(-d2 / (2.0 * sc ** 2))
    return out


def make_shadowing(bbox, n_bumps=40, std_db=7.0, scale=1500.0, rng=None):
    rng = rng or np.random.default_rng(1)
    (xmin, ymin, xmax, ymax) = bbox
    centers = np.column_stack([rng.uniform(xmin, xmax, n_bumps),
                               rng.uniform(ymin, ymax, n_bumps)])
    amps = rng.normal(0.0, std_db, n_bumps)
    scales = rng.uniform(0.6 * scale, 1.6 * scale, n_bumps)
    return dict(centers=centers, amps=amps, scales=scales)


def snr_dB(points, bs_xy, shadow, P_tx=46.0, G=15.0, N0=-96.0,
           PL0=32.0, n_exp=3.2, d0=1.0):
    """
    Best-server SNR (dB) at given points.
    Path loss PL(d)=PL0+10 n log10(max(d,d0)); RX = P_tx+G-PL-shadow; SNR=RX-N0.
    Shadowing is added on the *link* (per point), correlated in space.
    """
    pts = np.atleast_2d(points)
    sh = _shadow_field(pts, **shadow)               # (Npts,)
    best = np.full(pts.shape[0], -np.inf)
    for b in bs_xy:
        d = np.hypot(pts[:, 0] - b[0], pts[:, 1] - b[1])
        PL = PL0 + 10.0 * n_exp * np.log10(np.maximum(d, d0))
        rx = P_tx + G - PL - sh
        best = np.maximum(best, rx)
    return best - N0


# ----------------------------------------------------------------------
# 4. SNR -> service rate mu
# ----------------------------------------------------------------------
def snr_to_mu(snr_db, mu_max=14.0, snr_lo=2.0, snr_hi=22.0, mu_floor=0.0):
    """
    Smooth monotone SNR(dB) -> mu in [mu_floor, mu_max].
    Below snr_lo -> coverage gap (mu->mu_floor); above snr_hi -> mu_max.
    Uses a clipped Shannon-like shaping for a realistic knee.
    """
    x = np.clip((snr_db - snr_lo) / (snr_hi - snr_lo), 0.0, 1.0)
    shaped = np.log2(1.0 + 15.0 * x) / np.log2(1.0 + 15.0)   # concave knee
    return mu_floor + (mu_max - mu_floor) * shaped


def sample_mu_along_path(path_xy, s_path, bs_xy, shadow, speed_mps,
                         dt=0.05, mu_kwargs=None, prop_kwargs=None):
    """
    Train at constant speed: x(t)=speed*t along arc length.
    Returns t, mu_full(t) over the whole route (the realistic quasi-periodic profile).
    """
    mu_kwargs = mu_kwargs or {}
    prop_kwargs = prop_kwargs or {}
    total_s = s_path[-1]
    T_route = total_s / speed_mps
    t = np.arange(0.0, T_route, dt)
    s_q = speed_mps * t
    pos = point_at_arclength(path_xy, s_path, s_q)
    snr = snr_dB(pos, bs_xy, shadow, **prop_kwargs)
    mu = snr_to_mu(snr, **mu_kwargs)
    return t, mu, snr, pos


# ----------------------------------------------------------------------
# 5. Periodic component (quasi-periodic -> periodic via phase-folding)
# ----------------------------------------------------------------------
def periodic_component(t, mu_full, T0, n_phase=512):
    """
    Fold mu_full(t) modulo the dominant period T0 and average within phase bins
    -> the periodic idealisation mu_per(phase), phase in [0,T0).
    Returns phase grid (n_phase,), mu_per(n_phase,), and the residual std.
    """
    phase = np.mod(t, T0)
    edges = np.linspace(0.0, T0, n_phase + 1)
    idx = np.clip(np.searchsorted(edges, phase, side='right') - 1, 0, n_phase - 1)
    s = np.bincount(idx, weights=mu_full, minlength=n_phase)
    c = np.bincount(idx, minlength=n_phase).astype(float)
    mu_per = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    # fill empty bins by interpolation
    good = ~np.isnan(mu_per)
    ph = (edges[:-1] + edges[1:]) / 2.0
    mu_per = np.interp(ph, ph[good], mu_per[good])
    # residual (aperiodic part) magnitude
    mu_per_at_t = np.interp(phase, ph, mu_per)
    resid_std = float(np.std(mu_full - mu_per_at_t))
    return ph, mu_per, resid_std


def fourier_coeffs(mu_per, K):
    """
    a_m for m=-K..K from one period of mu_per (uniform samples).
    a[m] = (1/P) sum_p mu_per[p] exp(-i m w t_p) -> standard DFT scaling.
    Returns dict m->complex, with a[0] real = mean.
    """
    P = mu_per.size
    F = np.fft.fft(mu_per) / P            # F[0]=mean, F[m] for m=0..P-1
    a = {}
    for m in range(-K, K + 1):
        a[m] = F[m % P]
    return a


# ----------------------------------------------------------------------
# 6. Harmonic-balance spectral solver for periodic M/M(t)/1
# ----------------------------------------------------------------------
def solve_spectral(a, lam, omega, N, K):
    """
    Solve the truncated harmonic-balance system for c_{n,k}, n=0..N, |k|<=K.
    Kolmogorov: dp_n/dt = lam p_{n-1} - (lam+mu) p_n + mu p_{n+1}.
    Returns c (N+1, 2K+1) complex, with k-index offset by +K.
    """
    W = 2 * K + 1
    def idx(n, k): return n * W + (k + K)
    rows, cols, vals = [], [], []
    Ma = K
    def add(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)

    for n in range(N + 1):
        for k in range(-K, K + 1):
            r = idx(n, k)
            diag = 1j * k * omega + lam
            add(r, idx(n, k), diag)
            if n >= 1:
                add(r, idx(n - 1, k), -lam)                 # +lam p_{n-1}
            # + (mu p_n)_k = + sum_m a_m c_{n,k-m}
            # NB: state n=0 has NO service term (-mu p_0 absent in dp_0/dt),
            #     so this term is added only for n>=1.
            if n >= 1:
                for m in range(-Ma, Ma + 1):
                    km = k - m
                    if -K <= km <= K and abs(a[m]) > 0:
                        add(r, idx(n, km), a[m])
            # - (mu p_{n+1})_k  (drop at n=N truncation)
            if n <= N - 1:
                for m in range(-Ma, Ma + 1):
                    km = k - m
                    if -K <= km <= K and abs(a[m]) > 0:
                        add(r, idx(n + 1, km), -a[m])

    A = sparse.csr_matrix((vals, (rows, cols)), shape=((N + 1) * W, (N + 1) * W),
                          dtype=complex)
    b = np.zeros((N + 1) * W, dtype=complex)

    # Replace the (n,k)=(0,0) equation with the normalisation sum_n c_{n,0}=1.
    r0 = idx(0, 0)
    A = A.tolil()
    A.rows[r0] = [idx(n, 0) for n in range(N + 1)]
    A.data[r0] = [1.0 for _ in range(N + 1)]
    A = A.tocsr()
    b[r0] = 1.0

    c = spsolve(A, b)
    return c.reshape(N + 1, W)


def spectral_metrics(c, lam, omega, K, n_t=400, T=None):
    """From c: mean queue <Q>=q_0, cycle-mean delay <W>=q_0/lam, and Qbar(t)."""
    N = c.shape[0] - 1
    n = np.arange(N + 1)
    q = np.array([np.sum(n * c[:, k + K]) for k in range(-K, K + 1)])  # q_k
    Qmean = q[K].real                     # k=0
    Wmean = Qmean / lam
    if T is None:
        T = 2 * np.pi / omega
    t = np.linspace(0, T, n_t, endpoint=False)
    Qbar = np.real(np.sum([q[k + K] * np.exp(1j * k * omega * t)
                           for k in range(-K, K + 1)], axis=0))
    return dict(Qmean=Qmean, Wmean=Wmean, t=t, Qbar=Qbar, q=q)


# ----------------------------------------------------------------------
# 7. Event-driven Monte Carlo (thinning) with controllers
# ----------------------------------------------------------------------
def _mu_phase_interp(ph, mu_per, T0):
    """Return a function mu_per_of_t(t) = periodic interpolation of mu_per."""
    def f(t):
        return np.interp(np.mod(t, T0), ph, mu_per)
    return f


def run_mc(mu_of_t, lam, mu_max, B, controller, T_warm, T_run,
           Delay_max, rng, T0=None, ph=None, Qbar_pred=None,
           qagent=None, train=False, dt_sample=0.05, record_trace=False,
           fb_delay=0.0):
    """
    Event-driven thinning Monte Carlo for M(t)/M(t)/1/B with admission control.
    FIFO queue with per-packet timestamps -> *true* sojourn-time distribution
    (rigorous mean and 99th-percentile delay of accepted packets).
    controller: callable(Q_obs, t, ctx) -> alpha in [0,1].
    `fb_delay` > 0 models a control/measurement latency: feedback controllers
    (reactive, Q-learning) see the queue as it was `fb_delay` seconds ago, while
    the proactive controller uses the a-priori radio-map forecast (no lag).
    Returns dict of metrics over the window [T_warm, T_warm+T_run].
    """
    from collections import deque
    Lam = lam + mu_max + 1e-9
    t = 0.0
    Q = 0
    arr = deque()                          # arrival times of queued packets (FIFO)
    fbh = deque()                          # (t, Q) history for delayed observation
    offered = accepted = overflow = throttled = served = 0
    area_Q = 0.0
    t_meas0 = T_warm
    t_end = T_warm + T_run
    sojourns = []
    tr_t, tr_Q = [], []
    next_samp = t_meas0
    ctx = dict(T0=T0, ph=ph, Qbar_pred=Qbar_pred, Delay_max=Delay_max,
               B=B, lam=lam, mu_max=mu_max, qagent=qagent, rng=rng)

    def q_observed(tv, Qv):
        if fb_delay <= 0.0:
            return Qv
        fbh.append((tv, Qv))
        while len(fbh) >= 2 and fbh[1][0] <= tv - fb_delay:
            fbh.popleft()
        return fbh[0][1]

    def ql_state(Qv, tv):
        qa = ctx['qagent']
        qb = min(qa['nQ'] - 1, int(Qv / max(1, B) * qa['nQ']))
        pb = int((np.mod(tv, T0) / T0) * qa['nP']) % qa['nP'] if T0 else 0
        return qb, pb

    prev_sa = None
    act = 0
    while t < t_end:
        dt = rng.exponential(1.0 / Lam)
        seg_a = max(t, t_meas0); seg_b = min(t + dt, t_end)
        if seg_b > seg_a:
            area_Q += Q * (seg_b - seg_a)
        if record_trace:
            while next_samp < t + dt and next_samp < t_end:
                tr_t.append(next_samp); tr_Q.append(Q); next_samp += dt_sample
        t += dt
        if t >= t_end:
            break

        Qobs = q_observed(t, Q)
        if ctx['qagent'] is not None:
            qa = ctx['qagent']
            sb = ql_state(Qobs, t)
            if train and rng.random() < qa['eps']:
                act = int(rng.integers(qa['nA']))
            else:
                act = int(np.argmax(qa['Qtab'][sb[0], sb[1]]))
            alpha = qa['alphas'][act]
        else:
            alpha = controller(Qobs, t, ctx)

        mu_now = float(mu_of_t(t))
        u = rng.random() * Lam
        in_window = (t >= t_meas0)
        reward = 0.0
        if u < lam:                                   # offered arrival
            if in_window: offered += 1
            if rng.random() < alpha:                  # admitted by controller
                if Q < B:
                    Q += 1; arr.append(t)
                    if in_window: accepted += 1
                else:
                    if in_window: overflow += 1
                    reward -= 1.0
            else:
                if in_window: throttled += 1
                reward -= 0.40
        elif u < lam + mu_now:                         # service
            if Q > 0:
                Q -= 1; t_arr = arr.popleft(); served += 1
                if in_window: sojourns.append(t - t_arr)
        reward -= (Q / max(1.0, B))

        if ctx['qagent'] is not None and train:
            qa = ctx['qagent']; sb = ql_state(Q, t)
            if prev_sa is not None:
                ps, pa = prev_sa
                best_next = np.max(qa['Qtab'][sb[0], sb[1]])
                qa['Qtab'][ps[0], ps[1], pa] += qa['lr'] * (
                    reward + qa['gamma'] * best_next - qa['Qtab'][ps[0], ps[1], pa])
            prev_sa = (sb, act)

    Tw = T_run
    Qbar_meas = area_Q / Tw
    sj = np.array(sojourns, float)
    mean_delay = float(sj.mean()) if sj.size else 0.0
    p99_delay = float(np.percentile(sj, 99)) if sj.size else 0.0
    overflow_prob = overflow / max(1, offered)
    goodput = accepted / max(1, offered)
    throttle_rate = throttled / max(1, offered)
    return dict(Qbar=Qbar_meas, mean_delay=mean_delay, p99_delay=p99_delay,
                overflow_prob=overflow_prob, goodput=goodput,
                throttle_rate=throttle_rate,
                trace_t=np.array(tr_t), trace_Q=np.array(tr_Q))


# ---- controller factories -------------------------------------------------
def ctrl_droptail():
    return lambda Q, t, ctx: 1.0

def ctrl_static(alpha_s):
    return lambda Q, t, ctx: alpha_s

def ctrl_reactive(B, a_mid=0.7, a_lo=0.35):
    def f(Q, t, ctx):
        if Q < 0.3 * B: return 1.0
        if Q < 0.7 * B: return a_mid
        return a_lo
    return f

def ctrl_spectral(tau_lookahead, a_mid=0.7, a_lo=0.35):
    """Proactive: throttle based on the *forecast* backlog proxy at t+tau."""
    def f(Q, t, ctx):
        T0, ph, Qpred = ctx['T0'], ctx['ph'], ctx['Qbar_pred']
        if T0 is None or Qpred is None:
            return 1.0
        Wpred = np.interp(np.mod(t + tau_lookahead, T0), ph, Qpred) / ctx['lam']
        Dm = ctx['Delay_max']
        if Wpred > Dm: return a_lo
        if Wpred > 0.5 * Dm: return a_mid
        return 1.0
    return f

def ctrl_spectral_lookahead(mu_forecast_of_t, lam, H=20.0, n_h=40, gain=0.004, a_min=0.5):
    """
    Proactive controller using the *known* service profile (radio map) over a
    look-ahead horizon H. It throttles in proportion to the predicted service
    deficit  D(t)=\\int_t^{t+H} max(0, lam - mu(s)) ds, pre-draining the buffer
    before low-coverage stretches (periodic dips AND aperiodic coverage gaps).
    Floored at a_min to protect goodput.
    """
    hs = np.linspace(0.0, H, n_h)
    dh = hs[1] - hs[0]
    def f(Q, t, ctx):
        mu_ahead = mu_forecast_of_t(t + hs)
        deficit = np.sum(np.maximum(0.0, lam - mu_ahead)) * dh
        return float(np.clip(1.0 - gain * deficit, a_min, 1.0))
    return f


# ---- genuinely-spectral proactive controller (driven by p_n^*(t), not by mu) ----
def spectral_signals(c, lam, omega, K, B_th, T, n_t=400):
    """Periodic spectral predictors reconstructed from the harmonic-balance
    solution c over one period [0,T):
        W_proxy(t) = Qbar(t)/lam ,   tail_proxy(t) = sum_{n>=B_th} p_n^*(t),
    where p_n^*(t)=sum_k c_{n,k} e^{i k omega t}. These are the spectral
    queueing predictors used by the spectral admission controller."""
    N = c.shape[0] - 1
    t = np.linspace(0.0, T, n_t, endpoint=False)
    E = np.array([np.exp(1j * k * omega * t) for k in range(-K, K + 1)])  # (2K+1,n_t)
    P = np.empty((N + 1, n_t))
    for n in range(N + 1):
        P[n] = np.real((c[n, :][:, None] * E).sum(axis=0))                # p_n(t)
    P = np.clip(P, 0.0, None)
    Qbar = (np.arange(N + 1)[:, None] * P).sum(axis=0)
    tail = P[B_th:, :].sum(axis=0)
    return t, Qbar / lam, tail

def ctrl_spectral_metric(W_of_t, tail_of_t, lead=6.0, kW=0.8, kT=2.0,
                         Wmax=0.0, eps=0.0, a_min=0.65):
    """Genuinely-spectral proactive admission. The actuation kernel is the
    harmonic-balance solution c, read through its delay and tail predictors
    W_proxy(t)=Qbar(t)/lam and tail(t)=sum_{n>=B_th}p_n^*(t) (NOT the raw
    service profile). With a lead time `lead`, admission is reduced in
    proportion to the spectrally predicted excess about to occur,
        alpha(t)=clip(1 - kW[W_proxy(t+lead)-Wmax]_+ - kT[tail(t+lead)-eps]_+,
                      a_min, 1).
    Wmax and eps are taken as the period-means of W_proxy and tail, so the
    controller throttles above each load's own predicted-congestion mean and
    is load-adaptive; it pre-drains ahead of the predicted periodic peaks and
    admits fully at the predicted troughs (cell centres)."""
    def f(Q, t, ctx):
        W = float(W_of_t(t + lead)); Tl = float(tail_of_t(t + lead))
        return float(np.clip(1.0 - kW * max(0.0, W - Wmax)
                                 - kT * max(0.0, Tl - eps), a_min, 1.0))
    return f


# ---- combined proactive controller: spectral backbone + radio-map gaps -------
def ctrl_spectral_plus_map(W_of_t, tail_of_t, mu_forecast_of_t, lam,
                           lead=6.0, kW=0.8, kT=2.0, Wmax=0.0, eps=0.0, a_min=0.65,
                           H=16.0, gain=0.014, map_a_min=0.45):
    """Combined proactive admission: the genuinely-spectral controller
    (harmonic-balance predictors W_proxy, tail on the periodic backbone) AND a
    radio-map look-ahead over horizon H for the aperiodic coverage gaps.
    Admission is the more conservative (smaller alpha) of the two layers."""
    sm = ctrl_spectral_metric(W_of_t, tail_of_t, lead, kW, kT, Wmax, eps, a_min)
    rm = ctrl_spectral_lookahead(mu_forecast_of_t, lam, H, 40, gain, map_a_min)
    def f(Q, t, ctx):
        return min(sm(Q, t, ctx), rm(Q, t, ctx))
    return f

# ==============================================================================
# Section 3. Corridor geo-anchoring
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from pyproj import Transformer
from scipy import stats
from scipy.signal import argrelmin

# ---- locked operating point (see paper) ----
SPACING, NE = 3800.0, 3.45                 # cell spacing (m); path-loss exponent
PROP = dict(P_tx=43.0, n_exp=NE)
MU   = dict(mu_max=14.0, snr_lo=4.0, snr_hi=22.0, mu_floor=0.0)
B, DELAY_MAX, FB = 24, 0.6, 0.8            # buffer; target delay; feedback delay (s)
K, N = 22, 150                             # solver truncation
SEED = 7
SPEED = 300/3.6                            # 300 km/h -> m/s

# ---- Paris -> Milan trans-Alpine geo-anchor (real coordinates) ----
LAT0, LON0 = 48.8566,  2.3522              # origin terminus (Paris)
LAT1, LON1 = 45.4642,  9.1900             # destination terminus (Milan)
_MLAT = 111320.0
_T3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
def to_lonlat(x, y):
    lat = LAT0 + np.asarray(y)/_MLAT
    lon = LON0 + np.asarray(x)/(_MLAT*np.cos(np.radians(LAT0)))
    return lon, lat
def to_3857(x, y):
    lon, lat = to_lonlat(x, y); return _T3857.transform(lon, lat)
SEC = 1.0/np.cos(np.radians(LAT0))

# ---- route in the local metric frame: straight chord + gentle lateral wiggle ----
DXr = (LON1-LON0)*_MLAT*np.cos(np.radians(LAT0))
DYr = (LAT1-LAT0)*_MLAT
tau = np.linspace(0, 1, 9)
chord = np.array([DXr, DYr]); chord /= np.hypot(*chord)
perp = np.array([-chord[1], chord[0]])
wig = 6000*np.sin(2*np.pi*tau*1.5) + 3500*np.sin(2*np.pi*tau*3.0)
wp = np.column_stack([DXr*tau, DYr*tau]) + perp[None, :]*wig[:, None]
path, s_path = densify_polyline(wp, step=40.0)

# ---- base stations OFF the track at varying distances (both sides) ----
bs, s_bs = make_bs_layout(path, s_path, spacing=SPACING, jitter=0.04,
                          offset_mean=450., offset_jitter=350., drop_prob=0.0,
                          rng=np.random.default_rng(SEED))
pad = 4000
bbox = (path[:,0].min()-pad, path[:,1].min()-pad, path[:,0].max()+pad, path[:,1].max()+pad)
shadow_bb = make_shadowing(bbox, n_bumps=120, std_db=1.2, scale=2400., rng=np.random.default_rng(SEED))

# periodic backbone (no coverage gaps)
t, mu_bb, snr_bb, _ = sample_mu_along_path(path, s_path, bs, shadow_bb, SPEED, 0.05, MU, PROP)
T0 = SPACING/SPEED
ph, mu_per, resid = periodic_component(t, mu_bb, T0, n_phase=256)
mubar = float(mu_per.mean()); omega = 2*np.pi/T0
mu_per_of_t = _mu_phase_interp(ph, mu_per, T0)

# ---- five DEAD ZONES as coverage gaps / deep fades (NOT tunnels) ----
# placed at cell-edge troughs; depth (dB) and width (m) tuned to target severity:
#   z1 long full outage ~8 s, z2 ~5 s, z3 ~3 s, z4 partial mu~3.5, z5 partial mu~1.5
mins = argrelmin(mu_bb, order=200)[0]
fr = t[mins]/t[-1]
target_fr = [0.16, 0.34, 0.50, 0.67, 0.84]
chosen = [mins[np.argmin(np.abs(fr-tf))] for tf in target_fr]
dead_s = [t[i]*SPEED for i in chosen]
# (amp dB, scale m) per zone:
DEAD_AW = [(26.0, 198.0), (26.0, 112.0), (26.0, 70.0), (3.7, 110.0), (3.2, 110.0)]
DEADS = [(t[i]/t[-1], A, w) for i, (A, w) in zip(chosen, DEAD_AW)]   # (arclength frac, dB, m)
dead_xy_list = [point_at_arclength(path, s_path, s0) for s0 in dead_s]
shadow_full = dict(
    centers=np.vstack([shadow_bb['centers']] + [d[None, :] for d in dead_xy_list]),
    amps=np.append(shadow_bb['amps'],    [A for _, A, _ in DEADS]),
    scales=np.append(shadow_bb['scales'], [w for _, _, w in DEADS]))

# realized full profile (backbone + coverage gaps)
_, mu_full, snr_f, pos = sample_mu_along_path(path, s_path, bs, shadow_full, SPEED, 0.05, MU, PROP)
T_route = float(t[-1])
mu_full_of_t = lambda tt: np.interp(np.mod(tt, T_route+0.05), t, mu_full)
mu_peak = float(mu_full.max())
t_deads = [s0/SPEED for s0 in dead_s]

def ci95(samples):
    s = np.asarray(samples, float); n = len(s); mn = float(s.mean())
    if n < 2: return mn, 0.0
    return mn, float(stats.t.ppf(0.975, n-1) * s.std(ddof=1) / np.sqrt(n))

def ring_radius(snr_thr, P_tx=43, G=15, PL0=32, N0=-96, n_exp=NE):
    return 10**(((P_tx+G-PL0-N0)-snr_thr)/(10*n_exp))
r_g, r_y, r_o = ring_radius(18), ring_radius(10), ring_radius(4)

print(f"route {s_path[-1]/1000:.1f} km (Paris-Milan trans-Alpine chord) | {len(bs)} BS off-track | "
      f"v={SPEED*3.6:.0f} km/h | T0={T0:.1f} s | {len(DEADS)} coverage gaps")
print(f"mu_per: mean={mubar:.2f} min={mu_per.min():.2f} max={mu_per.max():.2f} resid={resid:.2f}")
print(f"coverage gaps @ {[round(x) for x in t_deads]} s  ({[round(s/1000) for s in dead_s]} km)")
below = mu_full < 0.5; ev=[]; i=0
while i < len(below):
    if below[i]:
        j=i
        while j<len(below) and below[j]: j+=1
        ev.append((j-i)*0.05); i=j
    else: i+=1
print(f"full-outage durations: {[round(x,1) for x in ev]} s | partial fades mu_min~3.5, 1.5")
print(f"coverage radii (m): strong={r_g:.0f} usable={r_y:.0f} weak={r_o:.0f} (cells overlap: {r_o>SPACING/2})")

# ==============================================================================
# Section 4. Static coverage map (terrain + rings + track)
# ==============================================================================
# Static coverage map: geo-anchored OVERVIEW (whole route) + zoomed INSET
# (hexagonal cells, off-track base stations, coverage gaps). Presented
# anonymously (no place labels). For a real backdrop in Colab, add a
# *label-free* basemap, e.g. contextily CartoDB Positron NoLabels in 3857.
from matplotlib.patches import RegularPolygon, Patch, Rectangle
from matplotlib.collections import PatchCollection
km = 1e-3
PX, PY = path[:,0]*km, path[:,1]*km
BX, BY = bs[:,0]*km, bs[:,1]*km
DXk = np.array([d[0] for d in dead_xy_list])*km
DYk = np.array([d[1] for d in dead_xy_list])*km

fig = plt.figure(figsize=(13.5, 7.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.0], wspace=0.16)
axO = fig.add_subplot(gs[0]); axI = fig.add_subplot(gs[1])

# ---- overview ----
axO.plot(PX, PY, '-', color='#2ecc40', lw=11, alpha=0.16, solid_capstyle='round', zorder=1)
axO.plot(PX, PY, '-', color='crimson', lw=2.3, solid_capstyle='round', zorder=4,
         label=f'HSR route (~{s_path[-1]/1000:.0f} km)')
axO.scatter(BX, BY, s=5, c='k', marker='^', zorder=3, label=f'base stations ({len(bs)})')
axO.scatter([PX[0],PX[-1]], [PY[0],PY[-1]], s=80, facecolor='white', edgecolor='k', lw=1.6, zorder=6)
axO.annotate("origin", (PX[0],PY[0]), xytext=(8,-15), textcoords='offset points', fontsize=9)
axO.annotate("terminus", (PX[-1],PY[-1]), xytext=(-6,12), textcoords='offset points', fontsize=9)
for i,(dx,dy) in enumerate(zip(DXk,DYk),1):
    axO.scatter([dx],[dy], s=130, facecolor='none', edgecolor='red', lw=2.0, zorder=7)
    axO.annotate(str(i),(dx,dy),xytext=(7,6),textcoords='offset points',
                 color='red',fontsize=11,fontweight='bold',zorder=8)
axO.plot([],[],'o',mfc='none',mec='red',mew=2,label=f'coverage gaps / deep fades (1-{len(DEADS)})')
s0 = DEADS[0][0]*s_path[-1]; win = (s_path>s0-18000)&(s_path<s0+18000)
wx0,wx1 = PX[win].min()-3, PX[win].max()+3; wy0,wy1 = PY[win].min()-3, PY[win].max()+3
axO.add_patch(Rectangle((wx0,wy0),wx1-wx0,wy1-wy0,fill=False,ec='navy',lw=1.4,ls='--',zorder=9))
axO.set_aspect('equal'); axO.set_xlabel("easting [km]"); axO.set_ylabel("northing [km]")
axO.set_title("(a) Geo-anchored HSR corridor: route, cells, coverage gaps")
axO.legend(loc='lower left', fontsize=8.5, framealpha=.93); axO.grid(alpha=.25)

# ---- inset (hexagonal cells) ----
selb = (BX>wx0-2)&(BX<wx1+2)&(BY>wy0-2)&(BY<wy1+2)
Rhex = (SPACING/2)*km*1.08
hexes = [RegularPolygon((cx,cy),numVertices=6,radius=Rhex,orientation=0) for cx,cy in zip(BX[selb],BY[selb])]
axI.add_collection(PatchCollection(hexes, facecolor='#2ecc40', alpha=0.24, edgecolor='#127a2e', lw=1.4, zorder=1))
for cx,cy in zip(BX[selb],BY[selb]):
    j = np.argmin((PX-cx)**2+(PY-cy)**2); axI.plot([cx,PX[j]],[cy,PY[j]], color='grey', lw=0.8, ls=':', zorder=3)
axI.plot(PX[win], PY[win], '-', color='crimson', lw=3.0, solid_capstyle='round', zorder=4, label='HSR route')
axI.scatter(BX[selb], BY[selb], s=50, c='k', marker='^', zorder=5, label='base stations (off-track)')
inw = (DXk>wx0)&(DXk<wx1)&(DYk>wy0)&(DYk<wy1)
axI.scatter(DXk[inw], DYk[inw], s=280, facecolor='none', edgecolor='red', lw=2.4, zorder=6)
for dx,dy in zip(DXk[inw],DYk[inw]):
    axI.annotate("gap 1",(dx,dy),xytext=(9,8),textcoords='offset points',color='red',fontsize=10,fontweight='bold')
axI.set_xlim(wx0,wx1); axI.set_ylim(wy0,wy1); axI.set_aspect('equal')
axI.set_xlabel("easting [km]"); axI.set_ylabel("northing [km]")
axI.set_title("(b) Inset (~36 km): hexagonal cells, off-track BSs, a coverage gap")
hh = [Patch(fc='#2ecc40', alpha=.45, ec='#127a2e', label='cell (hexagonal)')]
h0,l0 = axI.get_legend_handles_labels(); axI.legend(handles=h0+hh, loc='upper right', fontsize=8.5, framealpha=.93)
axI.grid(alpha=.25)
plt.tight_layout(); plt.savefig("coverage_static.png", dpi=150, bbox_inches="tight"); plt.close()

# ==============================================================================
# Section 5. Interactive map (folium)
# ==============================================================================
import folium
lon_r, lat_r = to_lonlat(path[:,0], path[:,1])
lon_b, lat_b = to_lonlat(bs[:,0],  bs[:,1])
m = folium.Map(location=[float(lat_r.mean()), float(lon_r.mean())], zoom_start=11,
               tiles="OpenTopoMap", attr="OpenTopoMap")
for la, lo in zip(lat_b, lon_b):
    for r, col in [(r_o, '#ff8c00'), (r_y, '#ffd400'), (r_g, '#2ecc40')]:
        folium.Circle([la, lo], radius=float(r), color=None, fill=True,
                      fill_color=col, fill_opacity=0.16, weight=0).add_to(m)
    folium.CircleMarker([la, lo], radius=3, color='black', fill=True, fill_opacity=1).add_to(m)
folium.PolyLine(list(zip(lat_r, lon_r)), color='crimson', weight=4,
                tooltip=f'HSR route v≈{SPEED*3.6:.0f} km/h').add_to(m)
for i, d in enumerate(dead_xy_list, 1):
    lo_d, la_d = to_lonlat(d[0], d[1])
    folium.Circle([float(la_d), float(lo_d)], radius=520, color='red', fill=False,
                  weight=2, dash_array='6', tooltip=f'coverage gap {i}').add_to(m)
m.save("coverage_map.html")  # interactive map: open in a browser

# ==============================================================================
# Section Relief basemap (contextily)
# ==============================================================================
import contextily as cx
# reproject local metres -> Web Mercator (EPSG:3857)
PX3, PY3 = to_3857(path[:,0], path[:,1])
BX3, BY3 = to_3857(bs[:,0],   bs[:,1])
gx, gy   = dead_xy_list[2]                      # one gap for the marker
GX3, GY3 = to_3857(np.array([gx]), np.array([gy]))

# window around this gap (widen/narrow as needed)
w = 9000
fig, ax = plt.subplots(figsize=(7,7))
ax.plot(PX3, PY3, color='crimson', lw=3, solid_capstyle='round', zorder=5)
ax.scatter(BX3, BY3, s=60, c='k', marker='^', zorder=6)
ax.scatter(GX3, GY3, s=260, facecolor='none', edgecolor='red', lw=2.4, ls='--', zorder=7)
ax.set_xlim(GX3[0]-w, GX3[0]+w); ax.set_ylim(GY3[0]-w, GY3[0]+w)
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
cx.add_basemap(ax, source=cx.providers.Esri.WorldShadedRelief,
               crs="EPSG:3857", attribution="Relief tiles \u00A9 Esri")
plt.savefig("coverage_basemap.png", dpi=150, bbox_inches="tight"); plt.close()
# Esri.WorldShadedRelief: grey relief, topo feel, no labels/borders;
# Esri.WorldHillshade: hillshaded relief, no labels;
# CartoDB.PositronNoLabels: clean light basemap, no labels (no relief).

# ==============================================================================
# Section 6. Validation of the spectral solver
# ==============================================================================
print("=== validation: spectral <Q> vs MC (periodic backbone, K=22) ===")
for rho in (0.5, 0.7, 0.85):
    lam = rho*mubar
    a = fourier_coeffs(mu_per, K)
    c = solve_spectral(a, lam, omega, N, K)
    sm = spectral_metrics(c, lam, omega, K, T=T0)
    mcs = [run_mc(mu_per_of_t, lam, mu_per.max(), B=400, controller=ctrl_droptail(),
                  T_warm=1000, T_run=12000, Delay_max=1e9,
                  rng=np.random.default_rng(400+10*s+int(100*rho)))['Qbar'] for s in range(4)]
    mcq = float(np.mean(mcs))
    print(f"  rho={rho:.2f}: spectral <Q>={sm['Qmean']:.3f} | MC <Q>={mcq:.3f} "
          f"(rel.err {abs(sm['Qmean']-mcq)/mcq*100:.1f}%)")

# ==============================================================================
# Section 7. Service profile and periodic regime
# ==============================================================================
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
mu_per_tiled = np.interp(np.mod(t, T0), ph, mu_per)
axs[0].plot(t, mu_per_tiled, lw=1.0, color='#d62728', ls='--', alpha=0.7, label=r'$\mu_{\rm per}$ (backbone, tiled)')
axs[0].plot(t, mu_full, lw=1.4, color='#1f77b4', label=r'$\mu_{\rm full}(t)$ (realised)')
axs[0].axhline(mubar, color='gray', ls=':', lw=1, label=r'$\bar\mu$')
thr_dip = 0.5*mubar
for j, td in enumerate(t_deads):
    win = (t > td-6) & (t < td+6); below = win & (mu_full < thr_dip)
    if below.any():
        axs[0].axvspan(t[below].min(), t[below].max(), color='red', alpha=0.13, label='coverage gaps' if j == 0 else None)
    axs[0].annotate(f'z{j+1}', xy=(td, 14.2), color='red', fontsize=8, ha='center')
axs[0].set_ylim(0, 15)
axs[0].set_xlabel('t, s'); axs[0].set_ylabel(r'$\mu$, pkt/s'); axs[0].legend(fontsize=8)
axs[0].set_title('Realized service rate: drops at each coverage gap')

lam = 0.8*mubar
a = fourier_coeffs(mu_per, K); c = solve_spectral(a, lam, omega, N, K)
sm = spectral_metrics(c, lam, omega, K, n_t=240, T=T0)
axs[1].plot(sm['t'], sm['Qbar'], lw=2.4, color='#d62728', label=r'spectral $\bar Q(t)$')
axs[1].axhline(sm['Qmean'], color='gray', ls=':', label=r'$\langle Q\rangle$')
axs[1].set_xlabel('t within period $T_0$, s'); axs[1].set_ylabel('mean queue')
axs[1].set_title(f'Periodic regime, $\\rho=0.8$ ($\\langle W\\rangle$={sm["Wmean"]:.3f}s)')
axs[1].legend(fontsize=9)
plt.tight_layout( ); plt.savefig("fig_07.png", dpi=150, bbox_inches="tight"); plt.close()

# ==============================================================================
# Section 8. Spectral controller on the periodic backbone (A')
# ==============================================================================
mu_per_of_t = _mu_phase_interp(ph, mu_per, T0)
B_TH = 16; LEAD, KW, KT, AMIN = 6.0, 0.8, 2.0, 0.65   # Wmax,eps = per-load means (load-adaptive)

def _spec_signals(lam):
    a = fourier_coeffs(mu_per, K); c = solve_spectral(a, lam, omega, N, K)
    tg, Ws, Ts = spectral_signals(c, lam, omega, K, B_TH, T0, 400)
    W  = lambda s: np.interp(np.mod(s, T0), tg, Ws)
    Ta = lambda s: np.interp(np.mod(s, T0), tg, Ts)
    return tg, Ws, Ts, W, Ta, float(Ws.mean()), float(Ts.mean())

def _runrep_per(ctrl, lam, R=12, fb=0.0):
    P, M, O, G = [], [], [], []
    for r in range(R):
        res = run_mc(mu_per_of_t, lam=lam, mu_max=float(mu_per.max()), B=B, controller=ctrl,
                     T_warm=80, T_run=700, Delay_max=DELAY_MAX,
                     rng=np.random.default_rng(2000+r*7), T0=T0, ph=ph, fb_delay=fb)
        P.append(res['p99_delay']); M.append(res['mean_delay'])
        O.append(res['overflow_prob']); G.append(res['goodput'])
    return dict(p99=ci95(P), W=ci95(M), ovf=ci95(O), gp=ci95(G))

rhos_p = [0.5, 0.7, 0.85]; resP = {'DropTail': [], 'Reactive': [], 'Spectral': []}
for rho in rhos_p:
    lam = rho*mubar
    tg, Ws, Ts, W, Ta, Wmax, eps = _spec_signals(lam)
    spec = ctrl_spectral_metric(W, Ta, LEAD, KW, KT, Wmax, eps, AMIN)
    resP['DropTail'].append(_runrep_per(ctrl_droptail(), lam))
    resP['Reactive'].append(_runrep_per(ctrl_reactive(B, a_mid=0.72, a_lo=0.35), lam, fb=FB))
    resP['Spectral'].append(_runrep_per(spec, lam))
    d = resP['Spectral'][-1]; rr = resP['Reactive'][-1]; dt = resP['DropTail'][-1]
    print(f"rho={rho}: Spectral p99={d['p99'][0]:.3f}+-{d['p99'][1]:.3f} | "
          f"Reactive {rr['p99'][0]:.3f} | DropTail {dt['p99'][0]:.3f} | gp(spec)={d['gp'][0]:.2f}")

lam85 = 0.85*mubar
tg, Ws, Ts, W, Ta, Wmax, eps = _spec_signals(lam85)
phg = np.linspace(0, T0, 400, endpoint=False)
alpha_phase = np.clip([1.0 - KW*max(0.0, W(x+LEAD)-Wmax) - KT*max(0.0, Ta(x+LEAD)-eps) for x in phg], AMIN, 1.0)

fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.3))
col = {'DropTail': '#7f7f7f', 'Reactive': '#2ca02c', 'Spectral': '#1f77b4'}
for k in ['DropTail', 'Reactive', 'Spectral']:
    y = [resP[k][i]['p99'][0] for i in range(3)]; e = [resP[k][i]['p99'][1] for i in range(3)]
    axs[0].errorbar(rhos_p, y, yerr=e, marker='o', lw=2.4 if k == 'Spectral' else 1.6, color=col[k], label=k, capsize=3)
axs[0].set_xlabel(r'$\rho$'); axs[0].set_ylabel('99th-pct delay, s')
axs[0].set_title('Periodic backbone (no coverage gaps)'); axs[0].legend(fontsize=9); axs[0].grid(alpha=.3)
ax = axs[1]; axr = ax.twinx()
ax.plot(phg, np.interp(np.mod(phg, T0), ph, mu_per), color='#d62728', lw=1.4, label=r'$\mu_{\rm per}(t)$')
ax.axhline(lam85, color='gray', ls=':', lw=1, label=r'$\lambda$')
axr.plot(phg, Ts, color='#9467bd', lw=1.6, label=r'tail proxy $\sum_{n\geq B_{\rm th}}p_n^*$')
axr.plot(phg, alpha_phase, color='#1f77b4', lw=2.2, label=r'$\alpha(t)$')
ax.set_xlabel('phase in $T_0$, s'); ax.set_ylabel(r'$\mu_{\rm per}$'); axr.set_ylabel(r'$\alpha$, tail proxy')
ax.set_title(r'Mechanism at $\rho=0.85$: $\alpha$ drops ahead of predicted tail peak')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, fontsize=7.5, loc='lower left')
plt.tight_layout(); plt.savefig("fig_08.png", dpi=150, bbox_inches="tight"); plt.close()

# ==============================================================================
# Section 9. Six-controller comparison vs load (full profile with coverage gaps)
# ==============================================================================
# Six-controller comparison over the FULL trans-Alpine profile (window = whole route).
# Featured: Spectral+map (genuinely-spectral backbone admission + radio-map
# look-ahead for the aperiodic coverage gaps). Goodput is reported in full:
# the spectral/proactive layer is latency-optimised and trades a little
# admitted load for lower delay (delay-vs-goodput, appropriate for delay-QoS).
rhos = np.array([0.5, 0.7, 0.85])
keys = ['DropTail', 'Static', 'Reactive', 'Radio-map', 'Spectral+map', 'Q-learning']
mets = ['mean_delay', 'p99_delay', 'overflow_prob', 'goodput']
B_TH = 16; LEAD, KW, KT, AMIN = 6.0, 0.8, 2.0, 0.65

def _spec_layers(lam):
    a = fourier_coeffs(mu_per, K); c = solve_spectral(a, lam, omega, N, K)
    tg, Ws, Ts = spectral_signals(c, lam, omega, K, B_TH, T0, 400)
    W  = lambda s: np.interp(np.mod(s, T0), tg, Ws)
    Ta = lambda s: np.interp(np.mod(s, T0), tg, Ts)
    return W, Ta, float(Ws.mean()), float(Ts.mean())

def run_all(rho, seed):
    lam = rho*mubar
    alpha_s = float(np.clip((mubar - 1.0/DELAY_MAX)/lam, 0.2, 1.0))
    W, Ta, Wmax, eps = _spec_layers(lam)
    common = dict(lam=lam, mu_max=mu_peak, B=B, Delay_max=DELAY_MAX,
                  T_warm=T_route, T_run=int(2*T_route), T0=T0, ph=ph)
    o = {}
    o['DropTail']     = run_mc(mu_full_of_t, controller=ctrl_droptail(),
                               rng=np.random.default_rng(seed+1), fb_delay=0.0, **common)
    o['Static']       = run_mc(mu_full_of_t, controller=ctrl_static(alpha_s),
                               rng=np.random.default_rng(seed+2), fb_delay=0.0, **common)
    o['Reactive']     = run_mc(mu_full_of_t, controller=ctrl_reactive(B, a_mid=0.72, a_lo=0.35),
                               rng=np.random.default_rng(seed+3), fb_delay=FB, **common)
    o['Radio-map']    = run_mc(mu_full_of_t,
                               controller=ctrl_spectral_lookahead(mu_full_of_t, lam, H=16, gain=0.014, a_min=0.45),
                               rng=np.random.default_rng(seed+4), fb_delay=0.0, **common)
    o['Spectral+map'] = run_mc(mu_full_of_t,
                               controller=ctrl_spectral_plus_map(W, Ta, mu_full_of_t, lam,
                                                                 LEAD, KW, KT, Wmax, eps, AMIN),
                               rng=np.random.default_rng(seed+5), fb_delay=0.0, **common)
    qa = dict(nQ=6, nP=8, nA=3, alphas=np.array([1.0, 0.7, 0.4]),
              Qtab=np.zeros((6, 8, 3)), eps=0.2, lr=0.2, gamma=0.9)
    run_mc(mu_full_of_t, controller=None, qagent=qa, train=True,
           rng=np.random.default_rng(seed+6), fb_delay=FB, **dict(common, T_warm=0, T_run=int(1.5*T_route)))
    qa['eps'] = 0.0
    o['Q-learning'] = run_mc(mu_full_of_t, controller=None, qagent=qa, train=False,
                             rng=np.random.default_rng(seed+7), fb_delay=FB, **common)
    return o

R = 4
res = {r: {k: {} for k in keys} for r in rhos}
for rho in rhos:
    acc = {k: {mm: [] for mm in mets} for k in keys}
    for rep in range(R):
        o = run_all(rho, seed=1000+int(100*rho)+rep*11)
        for k in keys:
            for mm in mets: acc[k][mm].append(o[k][mm])
    for k in keys:
        for mm in mets: res[rho][k][mm] = ci95(acc[k][mm])
    D = res[rho]['DropTail']; SM = res[rho]['Spectral+map']; RC = res[rho]['Reactive']
    pc = lambda a, b: (a-b)/a*100.0
    print(f"rho={rho}: Spectral+map W={SM['mean_delay'][0]:.3f} p99={SM['p99_delay'][0]:.3f} "
          f"ovf={SM['overflow_prob'][0]*100:.2f}% gp={SM['goodput'][0]:.3f} | "
          f"vs DropTail: W {pc(D['mean_delay'][0],SM['mean_delay'][0]):+.0f}% "
          f"p99 {pc(D['p99_delay'][0],SM['p99_delay'][0]):+.0f}% ovf {pc(D['overflow_prob'][0],SM['overflow_prob'][0]):+.0f}% | "
          f"vs Reactive: W {pc(RC['mean_delay'][0],SM['mean_delay'][0]):+.0f}% p99 {pc(RC['p99_delay'][0],SM['p99_delay'][0]):+.0f}%")

colors = {'DropTail':'#7f7f7f','Static':'#9467bd','Reactive':'#2ca02c',
          'Radio-map':'#17becf','Spectral+map':'#1f77b4','Q-learning':'#ff7f0e'}
panels = [('mean_delay','mean delay, s'), ('p99_delay','99th-pct delay, s'),
          ('overflow_prob','overflow probability'), ('goodput','goodput')]
fig, axs = plt.subplots(2, 2, figsize=(11, 8))
for ax,(key,lab) in zip(axs.ravel(), panels):
    for k in keys:
        ys = np.array([res[r][k][key][0] for r in rhos]); es = np.array([res[r][k][key][1] for r in rhos])
        ax.errorbar(rhos, ys, yerr=es, fmt='o-', color=colors[k], label=k, capsize=3,
                    lw=2.8 if k=='Spectral+map' else 1.4, ms=7 if k=='Spectral+map' else 5)
    ax.set_xlabel(r'$\rho=\lambda/\bar\mu$'); ax.set_ylabel(lab); ax.grid(alpha=.3)
axs[0,0].legend(fontsize=8, ncol=2)
fig.suptitle(f'Six-controller comparison over the full {s_path[-1]/1000:.0f} km profile '
             f'({len(DEADS)} coverage gaps; mean ± 95% CI, R={R})')
plt.tight_layout(); plt.savefig("fig_09.png", dpi=150, bbox_inches="tight"); plt.close()
print("Note: Spectral+map gives the lowest mean/p99 delay (proactive, latency-optimised) "
      "at a modest goodput cost vs Reactive -- a deliberate delay-vs-admitted-load trade-off for delay-bounded QoS.")

# ==============================================================================
# Section 10. Robustness across speed (180-420 km/h)
# ==============================================================================
# Robustness across speed: featured Spectral+map vs Reactive vs DropTail
# (full-route window per speed; new off-track layout consistent with cell 7).
kmh_list = [180, 250, 300, 360, 420]
skeys = ['DropTail', 'Reactive', 'Spectral+map']
sweep = {k: {'p99': [], 'p99e': [], 'ovf': [], 'ovfe': []} for k in skeys}
B_TH = 16; LEAD, KW, KT, AMIN = 6.0, 0.8, 2.0, 0.65
colors = {'DropTail':'#7f7f7f','Reactive':'#2ca02c','Spectral+map':'#1f77b4'}
def _layers_at(mp2, omega2, T02, lam):
    a = fourier_coeffs(mp2, K); c = solve_spectral(a, lam, omega2, N, K)
    tg, Ws, Ts = spectral_signals(c, lam, omega2, K, B_TH, T02, 400)
    return ((lambda s: np.interp(np.mod(s, T02), tg, Ws)),
            (lambda s: np.interp(np.mod(s, T02), tg, Ts)), float(Ws.mean()), float(Ts.mean()))
for kmh in kmh_list:
    SP = kmh/3.6
    bs2, _ = make_bs_layout(path, s_path, spacing=SPACING, jitter=0.04, offset_mean=450.,
                            offset_jitter=350., drop_prob=0., rng=np.random.default_rng(SEED))
    sh2 = make_shadowing(bbox, n_bumps=120, std_db=1.2, scale=2400., rng=np.random.default_rng(SEED))
    shf = dict(centers=np.vstack([sh2['centers']] + [d[None, :] for d in dead_xy_list]),
               amps=np.append(sh2['amps'], [a for _, a, _ in DEADS]),
               scales=np.append(sh2['scales'], [w for _, _, w in DEADS]))
    t2, mbb, _, _ = sample_mu_along_path(path, s_path, bs2, sh2,  SP, 0.05, MU, PROP)
    _, mf, _, _   = sample_mu_along_path(path, s_path, bs2, shf, SP, 0.05, MU, PROP)
    T02 = SPACING/SP; ph2, mp2, _ = periodic_component(t2, mbb, T02, n_phase=256)
    mb2 = mp2.mean(); omega2 = 2*np.pi/T02; Troute2 = float(t2[-1])
    mfo = lambda x, t2=t2, mf=mf: np.interp(np.mod(x, t2[-1]+0.05), t2, mf); mpk = float(mf.max())
    acc = {k: {'p99': [], 'ovf': []} for k in skeys}
    for rep in range(6):
        lam = 0.8*mb2; seed = 2000+rep*11
        W, Ta, Wmax, eps = _layers_at(mp2, omega2, T02, lam)
        common = dict(lam=lam, mu_max=mpk, B=B, Delay_max=DELAY_MAX, T_warm=Troute2, T_run=int(2*Troute2), T0=T02, ph=ph2)
        oD = run_mc(mfo, controller=ctrl_droptail(), rng=np.random.default_rng(seed+1), fb_delay=0.0, **common)
        oR = run_mc(mfo, controller=ctrl_reactive(B, a_mid=0.72, a_lo=0.35), rng=np.random.default_rng(seed+3), fb_delay=FB, **common)
        oS = run_mc(mfo, controller=ctrl_spectral_plus_map(W, Ta, mfo, lam, LEAD, KW, KT, Wmax, eps, AMIN), rng=np.random.default_rng(seed+5), fb_delay=0.0, **common)
        for k, ok in zip(skeys, [oD, oR, oS]):
            acc[k]['p99'].append(ok['p99_delay']); acc[k]['ovf'].append(ok['overflow_prob'])
    for k in skeys:
        mp, hp = ci95(acc[k]['p99']); mo, ho = ci95(acc[k]['ovf'])
        sweep[k]['p99'].append(mp); sweep[k]['p99e'].append(hp); sweep[k]['ovf'].append(mo); sweep[k]['ovfe'].append(ho)
    print(f"{kmh} km/h: " + " | ".join(f"{k} p99={sweep[k]['p99'][-1]:.2f} ovf={sweep[k]['ovf'][-1]*100:.1f}%" for k in skeys))

fig, axs = plt.subplots(1, 2, figsize=(12, 4.2)); km = np.array(kmh_list)
for k in skeys:
    p = np.array(sweep[k]['p99']); pe = np.array(sweep[k]['p99e'])
    o = np.array(sweep[k]['ovf'])*100; oe = np.array(sweep[k]['ovfe'])*100
    axs[0].plot(km, p, 'o-', color=colors[k], label=k, lw=2.6 if k=='Spectral+map' else 1.5)
    axs[0].fill_between(km, p-pe, p+pe, color=colors[k], alpha=0.18)
    axs[1].plot(km, o, 'o-', color=colors[k], label=k, lw=2.6 if k=='Spectral+map' else 1.5)
    axs[1].fill_between(km, o-oe, o+oe, color=colors[k], alpha=0.18)
axs[0].set_xlabel('speed, km/h'); axs[0].set_ylabel('99th-pct delay, s'); axs[0].set_title(r'Tail delay vs speed ($\rho=0.8$, mean $\pm$ 95% CI)'); axs[0].grid(alpha=.3); axs[0].legend(fontsize=9)
axs[1].set_xlabel('speed, km/h'); axs[1].set_ylabel('overflow, %'); axs[1].set_title('Overflow vs speed'); axs[1].grid(alpha=.3); axs[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig("fig_10.png", dpi=150, bbox_inches="tight"); plt.close()

# ==============================================================================
# Section 11. Proactivity mechanism: mean queue near the coverage gaps
# ==============================================================================
RHO_TR = 0.7
B_TH = 16; LEAD, KW, KT, AMIN = 6.0, 0.8, 2.0, 0.65
def _sm_factory(lam):
    a = fourier_coeffs(mu_per, K); c = solve_spectral(a, lam, omega, N, K)
    tg, Ws, Ts = spectral_signals(c, lam, omega, K, B_TH, T0, 400)
    W = lambda s: np.interp(np.mod(s, T0), tg, Ws); Ta = lambda s: np.interp(np.mod(s, T0), tg, Ts)
    return ctrl_spectral_plus_map(W, Ta, mu_full_of_t, lam, LEAD, KW, KT, float(Ws.mean()), float(Ts.mean()), AMIN)

def fold_trace(ctrl_factory, fb, rho, nseed=4, dt_s=0.2, ntrav=3):
    nb = int(T_route/dt_s); accum = np.zeros(nb); cnt = np.zeros(nb)
    for sd in range(nseed):
        lam = rho*mubar
        r = run_mc(mu_full_of_t, lam=lam, mu_max=mu_peak, B=B, controller=ctrl_factory(lam),
                   T_warm=2*T_route, T_run=ntrav*T_route, Delay_max=DELAY_MAX, T0=T0, ph=ph,
                   dt_sample=dt_s, record_trace=True, fb_delay=fb, rng=np.random.default_rng(700+sd))
        idx = np.clip((np.mod(r['trace_t'], T_route)/dt_s).astype(int), 0, nb-1)
        np.add.at(accum, idx, r['trace_Q']); np.add.at(cnt, idx, 1.0)
    return (np.arange(nb)+0.5)*dt_s, accum/np.maximum(cnt, 1)

gD, QD = fold_trace(lambda lam: ctrl_droptail(), 0.0, RHO_TR)
gR, QR = fold_trace(lambda lam: ctrl_reactive(B, a_mid=0.72, a_lo=0.35), FB, RHO_TR)
gS, QS = fold_trace(_sm_factory, 0.0, RHO_TR)
fig, ax = plt.subplots(figsize=(12, 4.4))
for j, td in enumerate(t_deads):
    ax.axvspan(td-1.5, td+1.5, color='red', alpha=0.14, label='coverage gaps' if j == 0 else None)
ax.plot(gD, QD, color='#7f7f7f', lw=1.6, label='DropTail (no control)')
ax.plot(gR, QR, color='#2ca02c', lw=1.6, label=f'Reactive (fb delay {FB}s)')
ax.plot(gS, QS, color='#1f77b4', lw=2.4, label='Spectral+map (proposed)')
ax.axhline(B, color='k', ls='--', lw=0.8, label=f'buffer B={B}')
ax.set_xlim(0, T_route); ax.set_ylim(0, B+2)
ax.set_xlabel('train position along route (time), s'); ax.set_ylabel(r'ensemble-mean $\langle Q\rangle$')
ax.set_title(f'Mean buffer occupancy along the route ($\\rho={RHO_TR}$): proactive pre-draining before every coverage gap')
ax.legend(fontsize=8, loc='upper left', ncol=2); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig_11.png", dpi=150, bbox_inches="tight"); plt.close()
print("per-zone peak <Q> (DropTail/Reactive/Spectral+map):")
for i, td in enumerate(t_deads, 1):
    mm = (gD > td-5) & (gD < td+6)
    print(f"  zone {i}: {QD[mm].max():.1f} / {QR[mm].max():.1f} / {QS[mm].max():.1f}  (B={B})")
print(f"route-mean <Q>: DropTail {QD.mean():.2f} | Reactive {QR.mean():.2f} | Spectral+map {QS.mean():.2f}")
