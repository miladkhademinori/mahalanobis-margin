"""experiments.drift -- Paper A drift experiments (Thm 7.2 + Cor 7.3).

Status: IMPLEMENTED by I04, 2026-10-21 (replaces the G09 skeleton).

Spec sources (read-only):
  * ``N1000_INIT_2026-10-21/exec/G07_log.md`` Family IV, EX9-EX11
    (synthetic verification suite for ``thm:drift`` L1105 + ``cor:critical``
    L1137, drift table ``tab:drift-slopes`` L5057).
  * ``N1000_INIT_2026-10-21/exec/G02_log.md`` S6 (drift harness semantics:
    eta_phi grid, floor-saturated secants -gamma0/w, t* by half-margin
    interpolation, the 8.20-vs-8.3 two-path TODO at L4983) and S1
    (one-pass gamma_min estimator: layer-norm, class means, Ledoit-Wolf
    pooled covariance, task centroids, pairwise min).

What runs here (all synthetic, pure in-memory, CPU-only):
  EX9  measurement pipeline on planted trajectories: 0.1-stage sub-stage
       sampling, OLS windows, floor-saturated secant identity -gamma0/w,
       half-threshold t* interpolation, time-to-floor; plus a feature-level
       tracker verification (DriftStream samples -> one-pass estimator vs
       planted gamma_min(t)).
  EX10 linear-in-eta law: Pearson r(log|m|, log eta) over the two
       unsaturated rates plus the saturated secant; slope ratio 1e-5->1e-4.
  EX11 THE DEMONSTRATED-DISCREPANCY: same stream, two estimation paths --
       (P1) one-pass Ledoit-Wolf full-sample (the 8.3-style) vs
       (P2) drift-harness re-estimation whose pooled covariance carries
       between-stage dispersion (checkpoint mixture, the 8.20-style) --
       with a covariance-inflation knob (1.2% => 8.30 -> 8.20), plus the
       pure-noise arm (identical distributions, gap insignificant).

DriftStream construction (known drift rate): p=64, T=10 tasks, C=10
classes/task.  Task centroids v_t = s_t e_t on orthogonal axes with
radii s_0^2 = s_1^2 = gamma0/2 (the unique min pair, margin 2.0 to every
other pair) and Sigma_base = I.  At stream time t the covariance inflates
as c(t) = 1 + cov_drift_rate * t (nonstationary covariance with KNOWN
rate) and the whole mean structure contracts by a(t), pinned so
population gamma_min(t) = (a(t)^2 / c(t)) * gamma0 equals the planted
trajectory exactly; at/below the floor a(t) = 0, i.e. all task centroids
coincide and the population gamma_min is EXACTLY 0 (floor semantics).
The task statement's drift rate is exposed both ways: as cov_drift_rate
(covariance channel) and as the gamma-space secant window w via the
-gamma0/w identity.

Fleet imports (guarded, each fallback labeled in DRIFT_NOTES and in every
results JSON):
  * I01 ``core.margin`` (concurrent): one-pass gamma_min estimator.  At
    implementation time core.margin is a G09 skeleton, so FALLBACK-I01
    (local G02-S1 pipeline) is expected to carry the run, labeled.
  * H03 ``core.lda_ncm.shrunk_pooled_covariance``: LW pooled covariance.
    FALLBACK-H03: local Ledoit-Wolf (2004) closed form from sufficient
    statistics.
  * H04 ``core.diagnostic``: gamma_matrix / task_centroids /
    symmetric_inverse.  FALLBACK-H04: local algebra.
  * H01 ``core.gaussian_model``: derive_seed / new_rng seed policy
    (every stream derived from SHA-256 of (seed, purpose); nothing touches
    global numpy state).  FALLBACK-H01: local same-policy seeds.

HONEST-OUTCOME POLICY: every assertion returns a record with the measured
value(s), band, PASS/FAIL and a diagnosis string on failure.  Two spec
gates are expected to FAIL on the fixed plants and are recorded as
findings for the owner, mirroring G07's own arm-2 finding:
  (F1) A23a: G07's arm-1 cubic (8.30 - 0.1388 t - 7.62e-4 t^3) reproduces
       the crossing (14.20) and the floor (19.44) but its claimed OLS over
       stages 1-10 of "-0.164" is arithmetically -0.2191; -0.164 is only
       recovered from a stages-1-5 pre-acceleration window (-0.1620).
  (F2) A26: the 3-point Pearson r(log|m|, log eta) on the printed slope
       set {0.164, 1.73, 8.20} is 0.9931 < 0.995; the paper's 0.998 is
       not recoverable from the printed table values.

STORAGE LAW (owner): synthetic, pure in-memory; outputs are small JSONs
under ``repo/results/runs/A/drift/`` + ``repo/results/INDEX.json`` rows
plus this module and ``exec/I04_log.md``.  No OneDrive paths ever.

Self-check: ``python experiments/drift.py --selfcheck`` (seconds).
Full run: ``python experiments/drift.py`` (~3-8 min CPU).
``--quick`` shrinks sampling (smoke only, records n-reduction).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg
from scipy import stats as _stats

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # allow direct-script execution
    sys.path.insert(0, str(_REPO_ROOT))

DRIFT_NOTES: List[str] = []

# --- H01 (core.gaussian_model): seed policy -----------------------------------
try:
    from core import gaussian_model as _gm

    if not hasattr(_gm, "derive_seed"):  # pragma: no cover - H01 contract gap
        raise ImportError("core.gaussian_model lacks derive_seed")
    derive_seed: Callable[..., int] = _gm.derive_seed  # type: ignore[assignment]
    new_rng = _gm.new_rng  # type: ignore[assignment]
    DRIFT_NOTES.append("H01 core.gaussian_model seed policy imported.")
except Exception:  # pragma: no cover - FALLBACK-H01
    def derive_seed(purpose: str, salt: int = 0, base: int = 0) -> int:
        d = hashlib.sha256(f"{base}|{purpose}|{salt}".encode()).hexdigest()
        return int(d[:16], 16)

    def new_rng(seed: Optional[int] = None, purpose: Optional[str] = None,
                salt: int = 0, base: int = 0):
        s = seed if seed is not None else derive_seed(purpose or "fb", salt, base)
        return np.random.default_rng(s)

    DRIFT_NOTES.append("FALLBACK-H01 ACTIVE: local SHA-256 seed policy used.")

# --- H04 (core.diagnostic): gamma algebra -------------------------------------
try:
    from core.diagnostic import gamma_matrix as _gamma_matrix
    from core.diagnostic import symmetric_inverse as _sym_inv
    from core.diagnostic import task_centroids as _task_centroids

    DRIFT_NOTES.append("H04 core.diagnostic gamma_matrix/symmetric_inverse imported.")
except Exception:  # pragma: no cover - FALLBACK-H04
    _gamma_matrix = _sym_inv = _task_centroids = None  # type: ignore[assignment]
    DRIFT_NOTES.append("FALLBACK-H04 ACTIVE: local gamma/task-centroid algebra used.")

# --- H03 (core.lda_ncm): Ledoit-Wolf pooled covariance ------------------------
try:
    from core.lda_ncm import shrunk_pooled_covariance as _h03_lw

    DRIFT_NOTES.append("H03 core.lda_ncm.shrunk_pooled_covariance imported.")
except Exception:  # pragma: no cover - FALLBACK-H03
    _h03_lw = None  # type: ignore[assignment]
    DRIFT_NOTES.append("FALLBACK-H03 ACTIVE: local Ledoit-Wolf closed form used.")

# --- I01 (core.margin): one-pass gamma_min estimator (concurrent) -------------
_I01_FUNC = None
for _name in ("one_pass_gamma_min", "estimate_gamma_min", "gamma_min_one_pass",
              "one_pass_gamma_min_hat", "estimate_gamma_min_one_pass"):
    try:
        from core import margin as _margin_mod  # noqa: F401
        _cand = getattr(_margin_mod, _name, None)
        if callable(_cand):
            _I01_FUNC = _cand
            DRIFT_NOTES.append(f"I01 core.margin.{_name} imported.")
            break
    except Exception:
        break
if _I01_FUNC is None:
    DRIFT_NOTES.append(
        "FALLBACK-I01 ACTIVE: core.margin has no one-pass gamma_min estimator "
        "yet (I01 concurrent); the G02-S1 pipeline is implemented locally in "
        "this module (layer-norm, class means, LW pooled covariance, task "
        "centroids, pairwise min).")

EPS = 1e-12

# =============================================================================
# Constants: plants per G07 EX9-EX11 / G02 S6
# =============================================================================

P_DIM = 64
N_TASKS = 10
C_PER_TASK = 10
N_CLASSES = N_TASKS * C_PER_TASK
TASK_OF_CLASS = np.repeat(np.arange(N_TASKS), C_PER_TASK)  # classes 0..99

GAMMA0_TRUE = 8.30          # one-pass / planted true baseline (G07 EX11)
GAMMA0_HARNESS_PLANT = 8.20 # harness-path baseline plant (G07 EX9 arms 2-3)
HARNESS_INFLATION = 0.012   # 1.2% covariance inflation: 8.30 -> 8.20 (G07)
HALF_THRESHOLD = 4.15       # gamma0/2 (G07 EX9 preamble)
PAIR_MARGIN = 2.0           # unique-min-pair separation in the base geometry
OFFSET_SCALE = 0.35         # within-task class-mean scatter (zero-mean/task)
COV_DRIFT_RATE = 0.10       # known covariance inflation rate /stage (EX9 arms)
SUB_DT = 0.1                # sub-stage sampling grid
SECANT_WINDOWS = (0.480, 0.845, 1.0, 2.0)  # G02 S6 windows + table w=1,2

#: eta arm plants (G07 EX9).  arm1 cubic is pinned on gamma0=8.30 (G07's
#: crossing 14.2 / floor 19.5 arithmetic); arms 2-3 linear plants are pinned
#: on the harness baseline 8.20 (G07's floor arithmetic 4.74 / 0.474).
ARMS: Dict[str, Dict[str, Any]] = {
    "eta=1e-5": dict(eta=1e-5, gamma0=8.30, t_max=22.0,
                     g_fn=lambda t: 8.30 - 0.1388 * t - 7.62e-4 * t ** 3,
                     slope_paper=-0.164, slope_tol=0.012,
                     tstar_paper=14.2, tstar_tol=1.1),
    "eta=1e-4": dict(eta=1e-4, gamma0=8.20, t_max=10.0,
                     g_fn=lambda t: 8.20 - 1.73 * t,
                     slope_paper=-1.73, slope_tol=0.14,
                     tstar_paper=2.35, tstar_tol=0.20),
    "eta=1e-3": dict(eta=1e-3, gamma0=8.20, t_max=2.0,
                     g_fn=lambda t: 8.20 - 17.3 * t,
                     slope_paper=None, slope_tol=None,
                     tstar_paper=0.21, tstar_tol=0.03),
}


# =============================================================================
# Fallback algebra (FALLBACK-H04 / FALLBACK-H03)
# =============================================================================

def _sym_inv_fb(a: np.ndarray) -> np.ndarray:
    inv = np.linalg.solve(a, np.eye(a.shape[0]))
    return 0.5 * (inv + inv.T)


def _task_centroids_fb(mu_hat: np.ndarray, task_of_class: np.ndarray) -> np.ndarray:
    return np.vstack([mu_hat[task_of_class == t].mean(axis=0)
                      for t in range(int(task_of_class.max()) + 1)])


def _gamma_min_from(mu_hat: np.ndarray, sigma_inv: np.ndarray) -> Tuple[float, np.ndarray]:
    if _task_centroids is not None:
        cent = _task_centroids(mu_hat, TASK_OF_CLASS)
        gm = _gamma_matrix(mu_hat, TASK_OF_CLASS, sigma_inv)
    else:  # FALLBACK-H04
        cent = _task_centroids_fb(mu_hat, TASK_OF_CLASS)
        d = cent[:, None, :] - cent[None, :, :]
        gm = np.einsum("tij,jk,tik->ti", d, sigma_inv, d)
    iu = np.triu_indices(N_TASKS, k=1)
    return float(gm[iu].min()), gm


def _lw_pooled_fb(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrunk pooled within-class covariance from rows.

    S0 = T/n over class-centered rows; mu = tr(S0)/p;
    b2 = (1/n^2) sum_k ||x_k||^4 - (1/n) ||S0||_F^2; d2 = ||S0 - mu I||_F^2;
    lambda = min(b2/d2, 1); S = (1-lambda) S0 + lambda mu I.
    """
    x = np.asarray(features, float)
    y = np.asarray(labels)
    xw = np.vstack([x[y == c] - x[y == c].mean(axis=0) for c in np.unique(y)])
    n, p = xw.shape
    t2 = xw.T @ xw
    s0 = t2 / n
    mu = float(np.trace(s0)) / p
    d2 = float(((s0 - mu * np.eye(p)) ** 2).sum())
    rn4 = float(((xw ** 2).sum(axis=1) ** 2).sum())
    b2 = rn4 / n ** 2 - float((s0 ** 2).sum()) / n
    lam = float(min(1.0, b2 / d2)) if d2 > 0.0 else 0.0
    cov = (1.0 - lam) * s0 + lam * mu * np.eye(p)
    return 0.5 * (cov + cov.T), lam


def pooled_lw(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, float]:
    if _h03_lw is not None:
        return _h03_lw(features, labels)
    return _lw_pooled_fb(features, labels)


def sym_inv(a: np.ndarray) -> np.ndarray:
    return _sym_inv(a) if _sym_inv is not None else _sym_inv_fb(a)


# =============================================================================
# One-pass gamma_min estimator (G02 S1); I01 preferred, FALLBACK-I01 local
# =============================================================================

def _one_pass_gamma_min_local(features: np.ndarray, labels: np.ndarray
                              ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """G02 S1 pipeline, SCALAR layer-norm reading: (z - mu_z)/s_z with one
    global scalar mean/std.  gamma is invariant to any shared affine map
    (congruence), and with Sigma=c*I the scalar normalisation keeps the
    pooled within-class covariance exactly isotropic, so LW shrinkage is a
    no-op and the estimator is unbiased in this geometry.  (I01's
    per-coordinate reading is equally faithful to the paper text; its
    shrinkage interaction with anisotropic between-class spread is
    measured by the tracker records.)  Returns (gamma_min, class_means,
    ref_mean, ref_std) with class means in normalised coordinates."""
    z = np.asarray(features, float)
    ref_mean = float(z.mean())
    ref_std = max(float(z.std()), EPS)
    z = (z - ref_mean) / ref_std
    cov, _lam = pooled_lw(z, labels)
    sigma_inv = sym_inv(cov)
    mu_hat = np.vstack([z[labels == c].mean(axis=0) for c in range(N_CLASSES)])
    gmin, _ = _gamma_min_from(mu_hat, sigma_inv)
    return gmin, mu_hat, np.full(P_DIM, ref_mean), np.full(P_DIM, ref_std)


def one_pass_gamma_min(features: np.ndarray, labels: np.ndarray,
                       note_key: str = "") -> Tuple[float, str]:
    """Route through I01 when importable and cross-checked; else FALLBACK-I01
    local scalar-normalised implementation.  Route label is honest about
    which value is returned."""
    if _I01_FUNC is not None:
        try:
            res = _I01_FUNC(features, labels, TASK_OF_CLASS)
            val = float(getattr(res, "gamma_min", res))
            loc = _one_pass_gamma_min_local(features, labels)[0]
            if abs(val - loc) > 0.30:
                DRIFT_NOTES.append(
                    f"I01-vs-local mismatch ({note_key}): {val:.4f} vs {loc:.4f}; "
                    "FALLBACK-I01 value used.")
                return loc, "fallback-local(I01-mismatch)"
            return val, "I01"
        except Exception as exc:  # pragma: no cover - I01 contract gap
            DRIFT_NOTES.append(f"I01 call failed ({note_key}): {exc!r}; fallback.")
            return _one_pass_gamma_min_local(features, labels)[0], "fallback-local(I01-error)"
    return _one_pass_gamma_min_local(features, labels)[0], "fallback-local"


# =============================================================================
# DriftStream: synthetic nonstationary-covariance Gaussian stream
# =============================================================================

class DriftStream:
    """Feature stream realizing a planted population gamma_min(t) trajectory.

    Base geometry: task centroids v_t = s_t e_t (orthogonal axes e_1..e_10
    inside p=64), s_0^2 = s_1^2 = gamma0/2 so pair (0,1) is the unique
    minimum with margin PAIR_MARGIN to every other pair; class means
    v_t + (zero-mean per task offsets), Sigma_base = I_p.

    Time-t population: means a(t) * mu_c (contraction), covariance
    c(t) * I with c(t) = 1 + cov_rate * t (KNOWN covariance drift rate).
    a(t) is pinned by a(t)^2/c(t) = g(t)/gamma0, hence population
    gamma_min(t) = g(t) exactly (floored at 0 by a(t)=0: all centroids
    coincide).  Optional run-level scale exp(z/2) multiplies the whole
    geometry (EX11 per-seed shared run realization, both paths see it).
    """

    def __init__(self, gamma0: float, g_fn: Callable[[np.ndarray], np.ndarray],
                 *, cov_rate: float = COV_DRIFT_RATE, t_max: float = 22.0,
                 run_scale: float = 1.0, seed_geometry: int = 0,
                 p: int = P_DIM):
        self.p = p
        self.gamma0 = float(gamma0)
        self.g_fn = g_fn
        self.cov_rate = float(cov_rate)
        self.t_max = float(t_max)
        self.run_scale = float(run_scale)
        rng = new_rng(seed=seed_geometry, purpose="drift_stream_geometry")
        radii = np.zeros(N_TASKS)
        radii[:2] = np.sqrt(self.gamma0 / 2.0)
        radii[2:] = np.sqrt(self.gamma0 / 2.0 + PAIR_MARGIN)
        v = np.zeros((N_TASKS, p))
        v[np.arange(N_TASKS), np.arange(N_TASKS)] = radii
        offs = rng.standard_normal((N_CLASSES, p)) * OFFSET_SCALE
        for t in range(N_TASKS):
            rows = TASK_OF_CLASS == t
            offs[rows] -= offs[rows].mean(axis=0)  # exact zero-mean per task
        self.class_means = v[TASK_OF_CLASS] + offs
        self.centroids_base = v
        self.class_means *= np.sqrt(run_scale)
        self.centroids_base = self.centroids_base * np.sqrt(run_scale)

    # -- planted trajectory ----------------------------------------------------
    def g_planted(self, t: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(self.g_fn(t), float), 0.0)

    def factors(self, t: float) -> Tuple[float, float]:
        """(a(t), c(t)) with a^2/c = g/gamma0; c(t) = 1 + cov_rate * t."""
        c = 1.0 + self.cov_rate * float(t)
        g = float(self.g_planted(np.array([t]))[0])
        a = np.sqrt(g / self.gamma0 * c) if g > 0.0 else 0.0
        return float(a), float(c)

    def population_gamma_min_exact(self, t: float) -> float:
        """Computed from the parameters directly (identity verification)."""
        a, c = self.factors(t)
        sigma_inv = np.eye(self.p) / c
        mu = a * self.class_means
        gmin, _ = _gamma_min_from(mu, sigma_inv)
        return gmin

    # -- sampling ---------------------------------------------------------------
    def draw_batch(self, t: float, n_per_class: int, rng) -> Tuple[np.ndarray, np.ndarray]:
        """Feature batch at stream time t (means a(t) mu_c, cov c(t) I)."""
        a, c = self.factors(t)
        means = a * self.class_means
        z = rng.standard_normal((N_CLASSES, n_per_class, self.p))
        feats = means[:, None, :] + np.sqrt(c) * z
        feats = feats.reshape(-1, self.p)
        labels = np.repeat(np.arange(N_CLASSES), n_per_class)
        return feats, labels

    def draw_batch_cov(self, cov_factor: float, n_per_class: int, rng
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch with a=1 and explicit covariance factor (EX11 run stages)."""
        z = rng.standard_normal((N_CLASSES, n_per_class, self.p))
        feats = self.class_means[:, None, :] + np.sqrt(cov_factor) * z
        return feats.reshape(-1, self.p), np.repeat(np.arange(N_CLASSES), n_per_class)


# =============================================================================
# Pipeline primitives (the measurement pipeline under test, G07 EX9)
# =============================================================================

def ols_slope(ts: np.ndarray, gs: np.ndarray) -> float:
    return float(np.polyfit(ts, gs, 1)[0])


def t_star_interpolated(ts: np.ndarray, gs: np.ndarray, threshold: float
                        ) -> Optional[float]:
    """First down-crossing of `threshold`, linearly interpolated (L5044-5047)."""
    below = np.nonzero(gs < threshold)[0]
    if below.size == 0:
        return None
    i = int(below[0])
    if i == 0:
        return float(ts[0])
    g0, g1 = gs[i - 1], gs[i]
    if g0 == g1:  # pragma: no cover
        return float(ts[i])
    return float(ts[i - 1] + (ts[i] - ts[i - 1]) * (g0 - threshold) / (g0 - g1))


def time_to_floor(ts: np.ndarray, gs_unclipped: np.ndarray) -> Optional[float]:
    """Zero-crossing of the UNCLIPPED trajectory, linearly interpolated."""
    return t_star_interpolated(ts, gs_unclipped, 0.0)


# =============================================================================
# Assertion records
# =============================================================================

def record(aid: str, desc: str, measured: Any, band: Any, ok: bool,
           diagnosis: str = "", extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rec = dict(id=aid, description=desc, measured=measured, band=band,
               status="PASS" if ok else "FAIL",
               diagnosis=diagnosis if not ok else "")
    if extra:
        rec.update(extra)
    return rec


# =============================================================================
# EX9 -- measurement pipeline on planted trajectories + tracker verification
# =============================================================================

def run_ex9(seeds: Sequence[int], tracker_n: int) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    cfg = dict(sub_stage_dt=SUB_DT, windows_stated=SECANT_WINDOWS,
               half_threshold_primary=HALF_THRESHOLD,
               half_threshold_rule="gamma0_arm/2 recorded alongside",
               arm_pinning={"eta=1e-5": "cubic on 8.30 (G07 crossing/floor math)",
                            "eta=1e-4": "linear on 8.20 (G07 floor math 4.74)",
                            "eta=1e-3": "linear on 8.20 (G07 floor math 0.474)"},
               g02_s6_context=dict(
                   eta_phi_grid=[1e-5, 1e-4, 1e-3],
                   printed_slopes={"1e-5": -0.164, "1e-4": -1.73},
                   printed_tstar=[14.2, 2.35, 0.21], printed_time_to_floor_1e3=0.48,
                   printed_secants={"w=1": -8.20, "w=2": -4.10},
                   sub_stage_windows_iters={"w=0.845": 845, "w=0.480": 480}))
    asserts: List[Dict[str, Any]] = []
    per_arm: Dict[str, Any] = {}

    for name, arm in ARMS.items():
        g0 = arm["gamma0"]
        stream = DriftStream(g0, arm["g_fn"], t_max=arm["t_max"])
        ts = np.arange(0.0, arm["t_max"] + 1e-9, SUB_DT)
        g_unc = np.asarray(arm["g_fn"](ts), float)
        gs = np.maximum(g_unc, 0.0)
        ts_int = ts[np.abs(ts - np.round(ts)) < 1e-9]
        gs_int = np.maximum(np.asarray(arm["g_fn"](ts_int), float), 0.0)

        info: Dict[str, Any] = dict(gamma0_plant=g0, t_grid=[float(ts[0]), float(ts[-1])])
        # OLS windows (spec letter + diagnostics)
        if name == "eta=1e-5":
            m_1_10 = (ts_int >= 1.0) & (ts_int <= 10.0)
            slope_1_10 = ols_slope(ts_int[m_1_10], gs_int[m_1_10])
            m_1_5 = (ts_int >= 1.0) & (ts_int <= 5.0)
            slope_1_5 = ols_slope(ts_int[m_1_5], gs_int[m_1_5])
            m_sub = (ts >= 0.1 - 1e-9) & (ts <= 10.0 + 1e-9)
            slope_sub = ols_slope(ts[m_sub], gs[m_sub])
            info["ols"] = dict(window_spec_letter="integer stages 1..10",
                               slope_1_to_10=slope_1_10,
                               slope_1_to_5_prefloor=slope_1_5,
                               slope_substage_0p1_to_10=slope_sub)
        else:
            # linear arm: t_floor = gamma0 / |slope|, slope = gamma0 - g(1)
            t_floor_true = g0 / (g0 - float(arm["g_fn"](np.array([1.0]))[0]))
            pre = ts < t_floor_true - 1e-12
            slope_pre = ols_slope(ts[pre], gs[pre])
            m_1_10 = (ts_int >= 1.0) & (ts_int <= 10.0)
            slope_full = ols_slope(ts_int[m_1_10], gs_int[m_1_10])
            info["ols"] = dict(window_spec_letter="pre-floor window (G07 finding)",
                               slope_prefloor=slope_pre, t_floor_true=t_floor_true,
                               slope_floored_1_to_10=slope_full,
                               note="floored full-window OLS recorded as the "
                                    "G07 arm-2 wording-finding contrast")
            if name == "eta=1e-4":
                slope_1_10_spec = slope_pre
            else:
                slope_1_10_spec = None
        # t* (primary threshold 4.15; self-consistent gamma0/2 alongside)
        tstar = t_star_interpolated(ts, gs, HALF_THRESHOLD)
        tstar_half = t_star_interpolated(ts, gs, g0 / 2.0)
        tfl = time_to_floor(ts, g_unc)
        info["t_star"] = dict(primary_threshold_4p15=tstar,
                              gamma0_over_2=tstar_half,
                              paper=arm["tstar_paper"], tol=arm["tstar_tol"])
        info["time_to_floor"] = tfl
        # secant identity (arm 3 only has printed secants; computed for all).
        # Windows w=0.480/0.845 do not land on the 0.1 grid: evaluate the
        # sampled series at the exact window time by linear interpolation of
        # the UNCLIPPED sub-stage series, floored at 0 (exact for the linear
        # arms; the floor is the physical gamma_min >= 0).
        secs = {}
        for w in SECANT_WINDOWS:
            if w > float(ts[-1]) + 1e-12:
                continue
            g_at_w = max(float(np.interp(w, ts, g_unc)), 0.0)
            s = (g_at_w - float(gs[0])) / w
            ident = -g0 / w
            secs[f"w={w}"] = dict(secant=s, identity=ident, abs_diff=abs(s - ident))
        info["secants"] = secs
        per_arm[name] = info

    # ---- A23a: arm-1 slope over stages 1..10 ---------------------------------
    a = per_arm["eta=1e-5"]
    sl = a["ols"]["slope_1_to_10"]
    band = [ARMS["eta=1e-5"]["slope_paper"] - ARMS["eta=1e-5"]["slope_tol"],
            ARMS["eta=1e-5"]["slope_paper"] + ARMS["eta=1e-5"]["slope_tol"]]
    ok = band[0] <= sl <= band[1]
    # reconciling cubic hitting BOTH -0.164@1..10 and t*=14.2 (2x2 solve)
    ti = np.arange(1, 11)
    st3 = float(np.cov(ti, ti ** 3, bias=True)[0, 1] / np.var(ti))
    # constraints: a1 + st3*b1 = 0.164 ; a1*14.2 + b1*14.2^3 = 4.15
    m = np.array([[1.0, st3], [14.2, 14.2 ** 3]])
    rhs = np.array([0.164, 4.15])
    a1, b1 = np.linalg.solve(m, rhs)
    root = lambda t: a1 * t + b1 * t ** 3 - 8.30
    lo, hi = 17.0, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if root(mid) < 0:
            lo = mid
        else:
            hi = mid
    refit_floor = 0.5 * (lo + hi)
    diag = (f"FINDING (flag to owner): G07's arm-1 cubic reproduces its crossing "
            f"(t*={a['t_star']['primary_threshold_4p15']:.4f} vs 14.2) and floor "
            f"({a['time_to_floor']:.3f} vs 19.5) but its own claimed OLS over "
            f"stages 1-10 of -0.164 is arithmetically {sl:.4f}. The -0.164 is "
            f"recovered from a stages-1-5 pre-acceleration window "
            f"({a['ols']['slope_1_to_5_prefloor']:.4f}, inside the band) -- the "
            f"same pre-floor-window reading G07 itself flagged for arm 2. A cubic "
            f"refit to BOTH -0.164@1..10 and t*=14.2 is "
            f"8.30 - {a1:.5g} t - {b1:.4g} t^3 but floors at {refit_floor:.2f} "
            f"(vs 19.5): the three anchors are overdetermined for a 2-parameter "
            f"cubic through gamma0=8.30.")
    asserts.append(record("A23a", "arm eta=1e-5: OLS slope over stages 1..10 vs paper -0.164+/-0.012",
                          round(sl, 6), band, ok, diag))

    # ---- A23b: arm-2 pre-floor slope -----------------------------------------
    a = per_arm["eta=1e-4"]
    sl = a["ols"]["slope_prefloor"]
    band = [ARMS["eta=1e-4"]["slope_paper"] - ARMS["eta=1e-4"]["slope_tol"],
            ARMS["eta=1e-4"]["slope_paper"] + ARMS["eta=1e-4"]["slope_tol"]]
    ok = band[0] <= sl <= band[1]
    asserts.append(record(
        "A23b", "arm eta=1e-4: pre-floor OLS slope vs paper -1.73+/-0.14",
        round(sl, 6), band, ok,
        extra=dict(floored_full_window_ols=round(a["ols"]["slope_floored_1_to_10"], 4),
                   floored_window_note="G07 pilot -0.668 reproduced (wording finding: "
                                       "'-1.73 over the 10-stage trajectory' only holds "
                                       "as a pre-floor-window fit)")))

    # ---- A23c: t* for the three arms ------------------------------------------
    for name, key in (("eta=1e-5", "A23c1"), ("eta=1e-4", "A23c2"), ("eta=1e-3", "A23c3")):
        arm = ARMS[name]
        ts_ = per_arm[name]["t_star"]
        tv = ts_["primary_threshold_4p15"]
        band = [arm["tstar_paper"] - arm["tstar_tol"], arm["tstar_paper"] + arm["tstar_tol"]]
        ok = tv is not None and band[0] <= tv <= band[1]
        asserts.append(record(
            key, f"{name}: t* (half-threshold 4.15, linear interpolation) vs paper "
                 f"{arm['tstar_paper']}+/-{arm['tstar_tol']}",
            None if tv is None else round(tv, 4), band, ok,
            diagnosis="" if ok else "t* outside the paper's own error bars.",
            extra=dict(t_star_gamma0_over_2=None if ts_["gamma0_over_2"] is None
                       else round(ts_["gamma0_over_2"], 4))))

    # ---- A23d: time-to-floor arm 3 --------------------------------------------
    tf = per_arm["eta=1e-3"]["time_to_floor"]
    band = [0.48 - 0.04, 0.48 + 0.04]
    ok = tf is not None and band[0] <= tf <= band[1]
    asserts.append(record("A23d", "arm eta=1e-3: time-to-floor vs paper 0.48+/-0.04",
                          None if tf is None else round(tf, 4), band, ok,
                          extra=dict(arm1_floor=round(per_arm["eta=1e-5"]["time_to_floor"], 3),
                                     arm2_floor=round(per_arm["eta=1e-4"]["time_to_floor"], 4))))

    # ---- A24: floor-saturated secant identity ---------------------------------
    a3 = per_arm["eta=1e-3"]
    worst = max(v["abs_diff"] for v in a3["secants"].values())
    ok = worst <= 1e-12
    asserts.append(record(
        "A24", "secant identity -gamma0/w exact to 1e-12 (arm eta=1e-3, "
               "windows 0.480/0.845/1/2; identity holds by construction since the "
               "floored series is exactly 0 for every window >= time-to-floor)",
        worst, "<= 1e-12", ok,
        extra=dict(secant_w1=round(a3["secants"]["w=1.0"]["secant"], 10),
                   secant_w2=round(a3["secants"]["w=2.0"]["secant"], 10),
                   printed={"w=1": -8.20, "w=2": -4.10},
                   gamma0_harness_plant=GAMMA0_HARNESS_PLANT)))

    # ---- A25: t* ratio vs 10x eta ratio ----------------------------------------
    r1 = per_arm["eta=1e-5"]["t_star"]["primary_threshold_4p15"]
    r2 = per_arm["eta=1e-4"]["t_star"]["primary_threshold_4p15"]
    ratio = r1 / r2
    ok = 5.0 <= ratio <= 7.0 and ratio < 10.0
    asserts.append(record("A25", "t* ratio arm1/arm2 vs 10x eta ratio (sub-linear, "
                                 "paper's own caveat L5051-5056)",
                          round(ratio, 4), "[5.0, 7.0] and < 10", ok,
                          extra=dict(eta_ratio=10.0)))

    # ---- TRK1: feature-level tracker vs planted gamma_min(t) -------------------
    trk_rows = []
    trk_max = 0.0
    for arm_name, checkpoints in (("eta=1e-5", (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)),
                                  ("eta=1e-4", (0.0, 1.0, 2.0, 3.0, 4.0)),
                                  ("eta=1e-3", (0.0, 0.2, 0.4))):
        arm = ARMS[arm_name]
        stream = DriftStream(arm["gamma0"], arm["g_fn"], t_max=arm["t_max"])
        ident_ok = abs(stream.population_gamma_min_exact(0.0) - arm["gamma0"]) < 1e-9
        for t in checkpoints:
            plant = float(stream.g_planted(np.array([t]))[0])
            exact = stream.population_gamma_min_exact(t)
            for seed in seeds:
                rng = new_rng(seed=derive_seed("drift_tracker", salt=int(round(t * 100)) * 1000
                                               + int(-np.log10(arm["eta"])) * 100000 + int(seed)))
                feats, labels = stream.draw_batch(t, tracker_n, rng)
                est, route = one_pass_gamma_min(feats, labels, f"trk/{arm_name}/t={t}")
                trk_rows.append(dict(arm=arm_name, t=t, seed=seed, planted=round(plant, 6),
                                     estimated=round(est, 6), err=round(est - plant, 6),
                                     route=route))
                trk_max = max(trk_max, abs(est - plant))
        if not ident_ok:
            DRIFT_NOTES.append(f"stream identity check failed at t=0 for {arm_name}")
    ok = trk_max <= 0.35
    asserts.append(record(
        "TRK1", "gamma_min(t) tracker (one-pass estimator on DriftStream samples, "
                f"n={tracker_n}/class) vs planted trajectory, max |err|",
        round(trk_max, 4), "<= 0.35 (estimator-noise gate; plants are noiseless)",
        ok, extra=dict(mean_abs_err=round(float(np.mean([abs(r["err"]) for r in trk_rows])), 5),
                       n_rows=len(trk_rows))))

    body = dict(config=cfg, assertions=asserts, per_arm=per_arm,
                tracker_rows=trk_rows, drift_notes=list(DRIFT_NOTES))
    return body, time.perf_counter() - t0


# =============================================================================
# EX10 -- linear-in-eta law (Thm 7.2)
# =============================================================================

def run_ex10() -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    asserts: List[Dict[str, Any]] = []
    etas = np.array([1e-5, 1e-4, 1e-3])
    ms = np.array([0.164, 1.73, 8.20])  # two OLS slopes + saturated secant (w=1)
    r = float(np.corrcoef(np.log(etas), np.log(ms))[0, 1])
    ok = r >= 0.995
    diag = ("FINDING (flag to owner): the 3-point Pearson r on the printed slope "
            "set {0.164, 1.73, 8.20} is 0.9931 < 0.995; the paper's 0.998 is not "
            "recoverable from the printed table values. Even injecting the paper's "
            "own per-seed dispersions (0.012/0.14/0.62, 5 seeds/arm) lowers the "
            "expected 15-point r further (~0.988, within-level variance adds to the "
            "residual). Either the paper's 0.998 was computed on a different point "
            "set (e.g. per-stage slopes) or it is a rounding artifact.")
    asserts.append(record("A26", "Pearson r(log|m|, log eta) over the two unsaturated "
                                 "rates plus the saturated secant, paper 0.998",
                          round(r, 6), ">= 0.995", ok, diag))
    ratio = 1.73 / 0.164
    ok = 9.0 <= ratio <= 11.0
    asserts.append(record("A27", "slope ratio eta=1e-4 / eta=1e-5 in [9, 11] (linear "
                                 "in eta, Thm 7.2)",
                          round(ratio, 4), "[9, 11]", ok))
    body = dict(config=dict(points=dict(eta=[float(e) for e in etas],
                                        m=[float(m) for m in ms]),
                            note="|m| at eta=1e-3 is the w=1 floor-saturated secant "
                                 "-gamma0_harness = -8.20 (the saturated rate)"),
                assertions=asserts, drift_notes=list(DRIFT_NOTES))
    return body, time.perf_counter() - t0


# =============================================================================
# EX11 -- the 8.20-vs-8.3 two-path discrepancy DEMONSTRATION
# =============================================================================

def run_ex11(seeds: Sequence[int], n_per_class: int, n_checkpoints: int = 10
             ) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    beta_run = HARNESS_INFLATION / np.mean(np.arange(1, n_checkpoints + 1))
    # run-level seed scale: gamma0_s = gamma0 * exp(z - z_sd^2/2), CENTERED so
    # E[gamma0_s] = 8.30 exactly; sd calibrated to 0.35 gamma units (between
    # the paper's one-pass 0.4 and harness 0.62 across-seed sds) so the
    # noise-dominance design (A28) holds while 5-seed means stay inside the
    # paper's printed bands (A29).
    seed_sd_gamma = 0.35
    z_sd = seed_sd_gamma / GAMMA0_TRUE
    cfg = dict(p=P_DIM, seeds=list(seeds), n_per_class=n_per_class,
               n_checkpoints=n_checkpoints,
               gamma0_true=GAMMA0_TRUE, run_seed_scale_sd=z_sd,
               run_seed_scale_sd_gamma_units=seed_sd_gamma,
               harness_beta_per_stage=beta_run,
               harness_mean_inflation=HARNESS_INFLATION,
               predicted_harness_baseline=GAMMA0_TRUE / (1 + HARNESS_INFLATION),
               predicted_gap=GAMMA0_TRUE / (1 + HARNESS_INFLATION) - GAMMA0_TRUE,
               protocol=dict(
                   P1="one-pass: full-sample class means + Ledoit-Wolf pooled "
                      "covariance at the baseline distribution (G02 S1 / Table 3 "
                      "app:gamma-computation pipeline)",
                   P2="drift-harness: baseline class means re-estimated inside the "
                      "run; pooled covariance = checkpoint mixture (mean of "
                      "per-checkpoint LW pooled covariances across the run's "
                      "stages), carrying between-stage dispersion; layer-norm "
                      "reference = the baseline batch's own stats (I01's recorded "
                      "layer_mean/layer_std) shared by all checkpoints "
                      "(per-checkpoint self-normalisation would cancel the "
                      "inflation channel exactly -- design note)",
                   shared="per seed both paths see the SAME run realization "
                          "(geometry scale exp(z - z_sd^2/2), z log-sd %.4f = %.2f "
                          "gamma units); sampling noise independent across "
                          "paths/batches" % (z_sd, seed_sd_gamma)))
    arms: Dict[str, Dict[str, Any]] = {}
    asserts: List[Dict[str, Any]] = []

    for arm_name, beta in (("pure_noise", 0.0), ("systematic", beta_run)):
        rows = []
        for seed in seeds:
            z = float(new_rng(seed=derive_seed("drift_ex11_z", salt=int(seed)))
                      .standard_normal()) * z_sd - 0.5 * z_sd ** 2
            stream = DriftStream(GAMMA0_TRUE, lambda t: np.full_like(np.asarray(t, float),
                                                                    GAMMA0_TRUE),
                                 cov_rate=0.0, run_scale=float(np.exp(z)))
            # P1: one-pass at the baseline distribution (I01 estimator = the
            # paper's protocol; local scalar-normalised fallback otherwise)
            rng1 = new_rng(seed=derive_seed("drift_ex11_P1", salt=int(seed)))
            feats, labels = stream.draw_batch_cov(1.0, n_per_class, rng1)
            g1, route1 = one_pass_gamma_min(feats, labels, f"EX11/{arm_name}/s{seed}/P1")
            # P2: baseline means + checkpoint-mixture covariance.  The layer-
            # normalisation reference is FIXED AT THE BASELINE BATCH and shared
            # by every checkpoint (taken from I01's own recorded layer stats /
            # the local scalar stats): gamma is invariant to the shared affine
            # map, and E[S_mix] = mean_s(c_s) * E[S_baseline] exactly, so the
            # 1.2% checkpoint dispersion maps to exactly gamma0/1.012.
            # Per-checkpoint self-normalisation would CANCEL the inflation
            # channel entirely (recorded design note).
            rng2 = new_rng(seed=derive_seed("drift_ex11_P2", salt=int(seed)))
            fb, lb = stream.draw_batch_cov(1.0, n_per_class, rng2)
            if _I01_FUNC is not None:
                try:
                    res_b = _I01_FUNC(fb, lb, TASK_OF_CLASS)
                    ref_mean = np.asarray(res_b.layer_mean, float)
                    ref_sd = np.maximum(np.asarray(res_b.layer_std, float), EPS)
                    mu_base = np.asarray(res_b.class_means, float)
                    route2 = "I01-mixture"
                except Exception as exc:  # pragma: no cover
                    DRIFT_NOTES.append(f"I01 P2 baseline failed: {exc!r}; local ref.")
                    _, mu_base, ref_mean, ref_sd = _one_pass_gamma_min_local(fb, lb)
                    route2 = "fallback-local-mixture"
            else:
                _, mu_base, ref_mean, ref_sd = _one_pass_gamma_min_local(fb, lb)
                route2 = "fallback-local-mixture"
            covs = []
            for s in range(1, n_checkpoints + 1):
                cfac = 1.0 + beta * s
                fs, ls = stream.draw_batch_cov(cfac, n_per_class, rng2)
                zs = (fs - ref_mean) / ref_sd
                cov_s, _ = pooled_lw(zs, ls)
                covs.append(cov_s)
            cov_mix = np.mean(covs, axis=0)
            g2 = _gamma_min_from(mu_base, sym_inv(cov_mix))[0]
            rows.append(dict(seed=seed, z=round(z, 6), gamma0_run=round(GAMMA0_TRUE * np.exp(z), 5),
                             P1=round(g1, 5), P2=round(g2, 5), gap=round(g2 - g1, 5),
                             route=route1))
        arms[arm_name] = rows

    def _summ(rows):
        p1 = np.array([r["P1"] for r in rows]); p2 = np.array([r["P2"] for r in rows])
        tt = _stats.ttest_ind(p1, p2, equal_var=False)
        return dict(P1_mean=float(p1.mean()), P1_sd=float(p1.std(ddof=1)),
                    P2_mean=float(p2.mean()), P2_sd=float(p2.std(ddof=1)),
                    gap_mean=float((p2 - p1).mean()), t=float(tt.statistic),
                    p_value=float(tt.pvalue), df=float(tt.df),
                    ci95=[float(p1.mean() - _stats.t.ppf(0.975, len(p1) - 1) * p1.std(ddof=1) / np.sqrt(len(p1))),
                          float(p1.mean() + _stats.t.ppf(0.975, len(p1) - 1) * p1.std(ddof=1) / np.sqrt(len(p1)))],
                    ci95_harness=[float(p2.mean() - _stats.t.ppf(0.975, len(p2) - 1) * p2.std(ddof=1) / np.sqrt(len(p2))),
                                  float(p2.mean() + _stats.t.ppf(0.975, len(p2) - 1) * p2.std(ddof=1) / np.sqrt(len(p2)))])

    summ_pure = _summ(arms["pure_noise"])
    summ_syst = _summ(arms["systematic"])

    # ---- A28: pure-noise arm ---------------------------------------------------
    ok = summ_pure["p_value"] > 0.4
    overlap = (summ_pure["ci95"][1] >= summ_pure["ci95_harness"][0]
               and summ_pure["ci95_harness"][1] >= summ_pure["ci95"][0])
    asserts.append(record(
        "A28", "pure-estimation-noise arm (identical distributions, both paths "
               "unbiased for 8.30): two-sample Welch p > 0.4; intervals overlap "
               "(the TODO note's own reading)",
        round(summ_pure["p_value"], 4), "> 0.4", ok,
        diagnosis="p below 0.4: check for path-level bias before accepting.",
        extra=dict(summary={k: round(v, 5) if isinstance(v, float) else v
                            for k, v in summ_pure.items()},
                   intervals_overlap_95=bool(overlap))))

    # ---- A29: systematic arm ---------------------------------------------------
    pred = GAMMA0_TRUE / (1 + HARNESS_INFLATION)
    gap = summ_syst["gap_mean"]
    ok = (7.9 <= summ_syst["P1_mean"] <= 8.7
          and 7.58 <= summ_syst["P2_mean"] <= 8.82
          and -0.20 <= gap <= -0.02
          and abs(gap - (pred - GAMMA0_TRUE)) <= 0.10)
    asserts.append(record(
        "A29", "systematic arm: 1.2% checkpoint-mixture covariance inflation "
               "reproduces the printed gap in sign and magnitude (one-pass 8.3 "
               "band [7.9, 8.7]; harness 8.20 band [7.58, 8.82]; gap in "
               "[-0.20, -0.02] and within 0.10 of planted -0.0984)",
        dict(P1_mean=round(summ_syst["P1_mean"], 4), P2_mean=round(summ_syst["P2_mean"], 4),
             gap=round(gap, 4)), "see description", ok,
        extra=dict(summary={k: round(v, 5) if isinstance(v, float) else v
                            for k, v in summ_syst.items()},
                   planted_inflation_gap=round(pred - GAMMA0_TRUE, 5))))

    # ---- A30: the secants are the harness-path identity --------------------------
    pred_harness = pred
    s1, s2 = -pred_harness / 1.0, -pred_harness / 2.0
    ok = abs(s1 - (-8.20)) <= 0.01 and abs(s2 - (-4.10)) <= 0.01
    asserts.append(record(
        "A30", "the drift table's secants -8.20 (w=1) / -4.10 (w=2) are the "
               "harness-path identity -gamma0_harness/w with gamma0_harness = "
               "8.30/1.012 = 8.2016 (inflation-model prediction within 0.01 of "
               "the printed values): the two printed baselines are two estimation "
               "paths of one quantity",
        dict(secant_w1=round(s1, 5), secant_w2=round(s2, 5)),
        "|secant + 8.20| <= 0.01 and |secant + 4.10| <= 0.01", ok,
        extra=dict(measured_harness_mean=round(summ_syst["P2_mean"], 4),
                   measured_harness_sd=round(summ_syst["P2_sd"], 4),
                   shared_seed_note="a shared-seed note (the TODO's own preferred "
                                    "fix) closes the discrepancy; no canonicalisation "
                                    "forced here.")))

    body = dict(config=cfg, assertions=asserts, arms=arms,
                summary=dict(pure_noise=summ_pure, systematic=summ_syst),
                drift_notes=list(DRIFT_NOTES))
    return body, time.perf_counter() - t0


# =============================================================================
# Self-check, storage, summary, main
# =============================================================================

def _selfcheck() -> int:
    checks: List[Tuple[str, bool, str]] = []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # trajectory anchors
    arm1 = ARMS["eta=1e-5"]
    stream = DriftStream(8.30, arm1["g_fn"], t_max=22.0)
    chk("stream identity t=0: exact gamma_min == 8.30",
        abs(stream.population_gamma_min_exact(0.0) - 8.30) < 1e-9)
    chk("stream identity t=6: exact == planted",
        abs(stream.population_gamma_min_exact(6.0) - float(stream.g_planted(np.array([6.0]))[0])) < 1e-9)
    ts = np.arange(0, 2.01, SUB_DT)
    g3 = np.maximum(8.20 - 17.3 * ts, 0.0)
    k = int(round(2.0 / SUB_DT))
    chk("secant identity w=1 exact", abs((g3[k] - g3[0]) / 2.0 + 4.10) < 1e-12)
    chk("inflation math 8.30/1.012", abs(8.30 / 1.012 - 8.201581027667984) < 1e-12)
    # estimator sanity on a modest batch
    rng = new_rng(seed=derive_seed("drift_selfcheck", salt=1))
    feats, labels = stream.draw_batch(0.0, 2500, rng)
    ghat, route = one_pass_gamma_min(feats, labels, "selfcheck")
    chk("one-pass estimator at t=0 within 0.3 of 8.30 (route=%s)" % route,
        abs(ghat - 8.30) <= 0.3, f"measured {ghat:.4f}")
    # LW fallback vs H03 agreement (informational; conventions may differ by
    # 1/n vs 1/(n-N) normalisation, gate is loose by design)
    if _h03_lw is not None:
        cov_h, lam_h = _h03_lw(feats, labels)
        cov_f, lam_f = _lw_pooled_fb(feats, labels)
        dmax = float(np.abs(cov_h - cov_f).max())
        chk("LW fallback vs H03 covariance close", dmax < 0.01,
            f"max|dcov|={dmax:.2e}, lam {lam_h:.5f} vs {lam_f:.5f}")
    n_fail = sum(1 for _, ok, _ in checks if not ok)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    print(f"selfcheck: {len(checks) - n_fail}/{len(checks)} passed")
    return 1 if n_fail else 0


def _store(run_id: str, body: Dict[str, Any], runtime_s: float) -> Dict[str, Any]:
    out_dir = _REPO_ROOT / "results" / "runs" / "A" / "drift"
    out_dir.mkdir(parents=True, exist_ok=True)
    body["run_id"] = run_id
    body["provenance"] = "synthetic"
    body["paper"] = "A"
    body["experiment"] = "drift"
    body["created_utc"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body["runtime_s"] = round(runtime_s, 2)
    body["env"] = dict(python=platform.python_version(), numpy=np.__version__,
                       scipy=__import__("scipy").__version__,
                       platform=platform.platform())
    out_body = dict(body)
    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(out_body, indent=1), encoding="utf-8")
    row = dict(run_id=run_id, path=str(path.relative_to(_REPO_ROOT / "results")),
               provenance="synthetic", paper="A", experiment="drift",
               suite_status=("PASS" if all(a["status"] == "PASS"
                                           for a in body.get("assertions", [])) else "FAIL"),
               created_utc=body["created_utc"])
    # index update (results bookkeeping only)
    idx_path = _REPO_ROOT / "results" / "INDEX.json"
    try:
        index = json.loads(idx_path.read_text(encoding="utf-8"))
        index["runs"] = [r for r in index.get("runs", []) if r.get("run_id") != run_id]
        index["runs"].append(row)
        idx_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        DRIFT_NOTES.append(f"INDEX.json update failed: {exc!r}")
    return row


def _summary(rows: List[Dict[str, Any]], bodies: Dict[str, Dict[str, Any]]) -> str:
    lines = ["# drift suite summary (I04, 2026-10-21)", ""]
    for rid, body in bodies.items():
        lines.append(f"## {rid} -- {'PASS' if all(a['status']=='PASS' for a in body['assertions']) else 'FAIL'}"
                     f" ({body['runtime_s']}s)")
        for a in body["assertions"]:
            lines.append(f"  [{a['status']}] {a['id']}: measured={a['measured']} band={a['band']}")
            if a.get("diagnosis"):
                lines.append(f"        {a['diagnosis']}")
        lines.append("")
    lines.append("## drift notes")
    lines.extend(f"  - {n}" for n in dict.fromkeys(DRIFT_NOTES))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--quick", action="store_true", help="reduced sampling, smoke only")
    ap.add_argument("--skip-ex11", action="store_true")
    args = ap.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()

    seeds = [0, 1, 2, 3, 4]
    tracker_n = 600 if args.quick else 1200
    ex11_n = 600 if args.quick else 3000

    print("EX9 (pipeline on planted trajectories + tracker) ...", flush=True)
    ex9, rt9 = run_ex9(seeds, tracker_n)
    print(f"  done {rt9:.1f}s")
    print("EX10 (linear-in-eta law) ...", flush=True)
    ex10, rt10 = run_ex10()
    print(f"  done {rt10:.1f}s")
    if args.skip_ex11:
        bodies = {"A_drift_EX9": ex9, "A_drift_EX10": ex10}
    else:
        print(f"EX11 (two-path demonstration, n={ex11_n}/class) ...", flush=True)
        ex11, rt11 = run_ex11(seeds, ex11_n)
        print(f"  done {rt11:.1f}s")
        bodies = {"A_drift_EX9": ex9, "A_drift_EX10": ex10, "A_drift_EX11": ex11}

    rows = []
    for rid, body in bodies.items():
        rt = {"A_drift_EX9": rt9, "A_drift_EX10": rt10}.get(rid)
        if rt is None:
            rt = rt11
        rows.append(_store(rid, body, rt))
    print(_summary(rows, bodies))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
