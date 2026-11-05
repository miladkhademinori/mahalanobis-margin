"""experiments.synthetic_margin -- Paper A G07 EX1-EX5 margin-theory suites.

Status: IMPLEMENTED by I03, 2026-10-21 (replaces the G09 skeleton).

Spec source (read-only): ``working/paper_A/exec/G07_log.md`` suites EX1-EX5
(Family I: Thm 3.1/3.2 infeasibility-feasibility, Thm 3.6 bound validity,
Cor 3.8 constant regimes, Thm 3.2-limit LDA check; Family II: Condition (S)
nu-ladder). EX6-EX8 (streaming) and EX9-EX11 (drift) belong to other
modules and are NOT touched here. Paper anchors (line numbers of
``working/paper_A/neurips_2026.tex`` as quoted by G07): Thm 3.1
``thm:infeasibility`` L505, Thm 3.2 ``thm:feasibility`` L542, Lemma 3.3
``lem:offdiag`` L683, Thm 3.6 ``thm:relaxed`` L710, Cor 3.8
``cor:operational`` L765, eq:xi-def L2948, eq:nu-def L2993, Condition (S)
L3027, eq:gamma-vs-spread L3042, Table-4 boundary rows L3142-3145, LDA
check L1227-1231 / L1441.

ARM POLICY (H05 lesson, binding): EXACT arms use strict tolerances
(float64 / solver precision); EMPIRICAL arms (Monte-Carlo, finite-sample
estimation) carry standard errors and SE-calibrated gates. EX1/EX3/EX5 and
the EX4 shared-statistics identity are EXACT; EX2 and the EX4
independent-estimation arm are EMPIRICAL (CRN, SE recorded).

OPERATIONAL DEFINITIONS (drift-recorded; see DRIFT_NOTES at runtime):
  * TC gap (EX1): full-objective excess F(w_CF) - min_w F(w) on random
    quadratic block losses; CF-optimum minimises the diagonal (within-task)
    sum only, exactly Thm 3.1's structure.
  * EX2 TC gap: Monte-Carlo zero-one error of the CF head minus the joint
    head on the planted Gaussian block construction (p=32, shared Sigma,
    simplex centroids so every task pair carries the same planted gamma;
    per-task softmax-calibrated CF head, joint = generative LDA/Bayes).
  * EX3 "TC gap" := E_TC, the cross-task confusion mass, whose gamma-decay
    Cor 3.8 predicts; computed EXACTLY (1-D erfc algebra + root finding),
    for both the joint head (primary gated arm; the aligned worst-pair law
    -(1-nu/2)^2/8 is exact for it) and the softmax-calibrated CF head
    (secondary arm). At nu=0 the calibrated head degenerates (coincident
    within-task classes); the joint arm is gated there, labeled.
  * EX4: CF-optimal generative LDA head vs joint LDA head, both from the
    SAME class statistics, evaluated with common random numbers: the heads
    are the identical function -> TC gap exactly 0.0 (bit-exact); the
    assertion floor < 1e-10 is therefore solver-limited (float64), exactly
    the paper's "attaining the feasibility limit" reading (L1227-1231).
  * EX5: planted nu-ladder with concentrated within-task deviation mass
    (one class per task at deviation D, rest at the centroid) so that
    max_k ||mu_k - mu^(t)||_* = sqrt(C tr Sigma_between^(t)) exactly; at
    the planted gamma this makes the constant-32 trace criterion of
    eq:gamma-vs-spread EXACTLY equivalent to the measured-nu <= 1/2
    criterion, and the constant-16 version exact for nu <= 1/sqrt(2) --
    which is A13's agreement design and A14's sqrt(2)-slack demo.

G07 SPEC DRIFT (found while implementing; all recorded at runtime):
  1. A6 tolerance: under the aligned geometry the exact exponent
     -(1-nu/2)^2/8 is steeper than the interpolating -(1-nu)/8 by
     nu^2/32, so the stated gate "slope >= -(1-nu)/8 - 0.005" can only
     hold for nu <= 0.4 (G07's own pilot numbers fail it at nu=0.5/0.73).
     Kept as stated (honest FAIL with diagnosis); a corrected pointwise
     ratio-validity assertion A6b is added, which is the mathematically
     right reading of "the interpolating form is a valid bound".
  2. A8 bracket [e^{-g/8}/1.2, 1.2 e^{-g/16}] is only applicable where
     the exact exponent lies in [1/16, 1/8], i.e. nu <= 2-sqrt(2); gated
     there, boundary rows recorded as exceeding it (consistent with the
     interpolating-form reading of Cor 3.8).
  3. A12: G07's planted ladder contains THREE rows > 1/2 (0.51, 0.586,
     0.73), so the planted operational count is 9/12, not "10 of 12"
     (that count mirrors the paper's real Table 4, L3142).
  4. EX2 bound prefactor M C^2 T(T-1)/(N-1) = 90.9 at spec parameters
     exceeds 1 for gamma <= 48, so A3 as stated is non-binding there;
     verified as stated AND with normalised margins recorded.

Fleet imports (guarded, each fallback labeled in DRIFT_NOTES):
  * I01 ``core.margin`` (concurrent): probed for the gamma/nu/Condition-(S)
    products; at run time it was a G09 skeleton -> FALLBACK-I01 local
    algebra (verified against H04) used. If I01 lands, its functions are
    preferred automatically (attribute probe).
  * I02 ``core.streaming`` (concurrent): probed for provenance only
    (EX1-EX5 need no streaming); skeleton noted.
  * H01 ``core.gaussian_model``: derive_seed/new_rng seed policy.
  * H04 ``core.diagnostic``: gamma_ij (used as the exact-arm gamma oracle).

STORAGE LAW (owner): synthetic, pure in-memory; outputs are small JSONs
under ``repo/results/runs/A/synthetic_margin/`` (provenance "synthetic",
G08 S2 envelope) plus INDEX.json entries and a markdown summary on stdout
(embedded verbatim in ``working/paper_A/exec/I03_log.md``). No OneDrive.

Self-check: ``python experiments/synthetic_margin.py --selfcheck`` (fast
identity checks). Full run: ``python experiments/synthetic_margin.py``
(EX1 <5s, EX2 ~4-6 min, EX3 <10s, EX4 <1 min, EX5 <1 min; reductions, if
any, are labeled in the JSON config and the summary).
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg
from scipy.optimize import brentq
from scipy.special import erfc, logsumexp

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # allow direct-script execution
    sys.path.insert(0, str(_REPO_ROOT))

DRIFT_NOTES: List[str] = []
_SUITE = "synthetic_margin"

# --- H01 (core.gaussian_model): conftest seed policy --------------------------------
try:  # pragma: no cover - normal path
    from core import gaussian_model as _gm

    derive_seed = _gm.derive_seed
    new_rng = _gm.new_rng
    DRIFT_NOTES.append("H01 core.gaussian_model imported (seed policy reused).")
except ImportError:  # pragma: no cover - FALLBACK-seeds
    import hashlib as _hl

    def derive_seed(purpose: str, salt: int = 0) -> int:  # type: ignore[misc]
        d = _hl.sha256(f"0|{purpose}|{salt}".encode()).hexdigest()
        return int(d[:16], 16)

    def new_rng(seed: Optional[int] = None, purpose: Optional[str] = None,
                salt: int = 0):  # type: ignore[misc]
        s = seed if seed is not None else derive_seed(purpose or "fallback", salt)
        return np.random.default_rng(s)

    DRIFT_NOTES.append(
        "FALLBACK-SEEDS ACTIVE: core.gaussian_model missing; local SHA-256 "
        "purpose seeding used (same (SEED=0, purpose) scheme).")

# --- H04 (core.diagnostic): gamma_ij ------------------------------------------------
try:  # pragma: no cover - normal path
    from core.diagnostic import gamma_ij as _h04_gamma_ij

    DRIFT_NOTES.append("H04 core.diagnostic.gamma_ij imported (exact-arm oracle).")
except ImportError:  # pragma: no cover - FALLBACK-H04
    _h04_gamma_ij = None
    DRIFT_NOTES.append("FALLBACK-H04 ACTIVE: local gamma_ij algebra used.")

# --- I01 (core.margin, concurrent) ---------------------------------------------------
try:  # pragma: no cover - probe
    from core import margin as _margin_mod

    _MARGIN_DOC = (getattr(_margin_mod, "__doc__", "") or "")[:200]
    if "NOT_IMPLEMENTED_YET" in _MARGIN_DOC or "SKELETON" in _MARGIN_DOC.upper():
        DRIFT_NOTES.append(
            "I01 core.margin present but SKELETON at I03 run time -> "
            "FALLBACK-I01 local gamma/nu/Condition-(S) algebra used "
            "(cross-checked against H04 gamma_ij in --selfcheck).")
        _margin = {}
    else:
        _margin = {name: getattr(_margin_mod, name) for name in
                   ("gamma_ij_pairwise", "nu_spread_ratio", "e_tc_bound",
                    "gamma_min_population", "block_loss_matrix", "tc_gap")
                   if hasattr(_margin_mod, name)}
        DRIFT_NOTES.append(
            "I01 core.margin imported; using: %s (local algebra = labeled "
            "fallback only)." % (sorted(_margin) or "(none; fallback used)"))
except ImportError:  # pragma: no cover - FALLBACK-I01
    _margin = {}
    DRIFT_NOTES.append(
        "FALLBACK-I01 ACTIVE: core.margin missing; local algebra used.")

# --- I02 (core.streaming, concurrent): provenance only -------------------------------
try:  # pragma: no cover - probe
    from core import streaming as _streaming_mod  # noqa: F401

    _DOC = (getattr(_streaming_mod, "__doc__", "") or "")[:200]
    state = ("SKELETON at I03 run time (EX1-EX5 need no streaming; noted for "
             "provenance)" if "NOT_IMPLEMENTED_YET" in _DOC or
             "SKELETON" in _DOC.upper() else "implemented")
    DRIFT_NOTES.append(f"I02 core.streaming present: {state}.")
except ImportError:  # pragma: no cover - FALLBACK-I02
    DRIFT_NOTES.append(
        "FALLBACK-I02 NOTE: core.streaming missing at I03 run time (EX1-EX5 "
        "do not use it; no fallback needed).")

# --- G07 spec drift found while implementing (see module docstring) ---------------
DRIFT_NOTES.extend([
    "G07-A6 drift: aligned-geometry exact exponent -(1-nu/2)^2/8 is steeper "
    "than the interpolating -(1-nu)/8 by nu^2/32, so the stated 0.005 "
    "tolerance can only hold for nu <= 0.4 (G07's own pilots imply this); "
    "kept as stated with honest FAIL + corrected pointwise ratio assertion.",
    "G07-A8 drift: the doubling bracket's 1.2 slack is smaller than the "
    "Gaussian-tail prefactor factor ~2^{-1/4} per doubling; corrected "
    "local-exponent bracket gated alongside.",
    "G07-A12 drift: the planted 12-row ladder has THREE rows > 1/2 "
    "(0.51, 0.586, 0.73), so the operational count is 9/12, not 10/12 "
    "(the 10/12 figure mirrors the paper's real Table 4, L3142).",
    "G07-EX3 pattern: G07's pilot slopes are reproduced exactly by the "
    "two-point aligned pattern (C/2 classes at +-s_max); a jittered "
    "full-spread pattern adds sibling-class transient contamination up to "
    "0.014 at nu=0.586 over [16,256]. Two-point adopted (deterministic).",
    "EX2/EX3 head-class finding: the per-task softmax-calibrated head has an "
    "x-dependent bias (outside ass:ncm-head) and its cross-task confusion "
    "does NOT decay with the margin (E_TC -> 1, slope ~ +1e-4); it is run "
    "as a labeled extension arm; the in-scope generative CF head attains "
    "gap exactly 0 (feasibility limit) -- consistent with Remark on "
    "generative methods (L652-663) and Remark 4.5 scoping.",
])


# =============================================================================
# Local exact algebra (FALLBACK-I01; used unless core.margin provides the name)
# =============================================================================


def gamma_ij(mu_i: np.ndarray, mu_j: np.ndarray, cov: np.ndarray) -> float:
    """eq:gamma L587 (Def 3.1): (mu_i-mu_j)' ((Sigma_i+Sigma_j)/2)^{-1}
    (mu_i-mu_j). Shared-covariance model: the pair average of equal
    matrices is the matrix. Prefers I01 ``gamma_ij_pairwise``."""
    fn = _margin.get("gamma_ij_pairwise")
    if fn is not None:
        return float(fn(mu_i, mu_j, cov, cov))
    d = np.asarray(mu_i, float) - np.asarray(mu_j, float)
    return float(d @ np.linalg.solve(np.asarray(cov, float), d))


def nu_ij(means: np.ndarray, task_of_class: np.ndarray, i: int, j: int,
          cov: np.ndarray) -> float:
    """eq:nu-def L2993 with the *-norm of the pair metric (Def 3.1); with a
    shared Sigma the pair metric IS Sigma^{-1}. Prefers I01
    ``nu_spread_ratio``; local algebra is the labeled fallback (identical
    for shared Sigma; cross-checked in --selfcheck)."""
    fn = _margin.get("nu_spread_ratio")
    if fn is not None:
        return float(fn(means, cov, task_of_class, i, j))
    si = np.linalg.inv(np.asarray(cov, float))
    d_i = means[task_of_class == i] - means[task_of_class == i].mean(axis=0)
    d_j = means[task_of_class == j] - means[task_of_class == j].mean(axis=0)
    mi = float(np.sqrt(max(np.max(np.einsum("ri,ij,rj->r", d_i, si, d_i)), 0.0)))
    mj = float(np.sqrt(max(np.max(np.einsum("ri,ij,rj->r", d_j, si, d_j)), 0.0)))
    c_i = means[task_of_class == i].mean(axis=0)
    c_j = means[task_of_class == j].mean(axis=0)
    g = gamma_ij(c_i, c_j, cov)
    return float((2.0 * mi + 2.0 * mj) / np.sqrt(g))


def e_tc_bound(gamma_min_value: float, M: float, C: int, T: int, N: int,
               kappa: float = 1.0 / 16.0) -> float:
    """Thm 3.6 bound M C^2 T(T-1)/(N-1) exp(-kappa gamma_min). Prefers I01
    ``e_tc_bound`` (rate 'conservative' = kappa 1/16)."""
    fn = _margin.get("e_tc_bound")
    if fn is not None and abs(kappa - 1.0 / 16.0) < 1e-15:
        return float(fn(gamma_min_value, M=M, C=C, T=T, N=N,
                        rate="conservative"))
    return float(M * C * C * T * (T - 1) / (N - 1)
                 * np.exp(-kappa * gamma_min_value))


def gamma_min_pop(means: np.ndarray, cov: Optional[np.ndarray],
                  task_of_class: np.ndarray) -> float:
    """Def 3.1 gamma_min from class-level parameters. Prefers I01."""
    fn = _margin.get("gamma_min_population")
    if fn is not None:
        return float(fn(means, cov, task_of_class).gamma_min)
    cents = np.vstack([means[task_of_class == t].mean(axis=0)
                       for t in range(int(task_of_class[-1]) + 1)])
    T = cents.shape[0]
    return min(gamma_ij(cents[i], cents[j], cov if cov is not None else np.eye(cents.shape[1]))
               for i in range(T) for j in range(i + 1, T))


def condition_s_trace(gamma: float, tr_between_i: float, tr_between_j: float,
                      C: int, sigma_min_sq: float, constant: float) -> bool:
    """eq:gamma-vs-spread trace criterion at ``constant`` in {16, 32}:
    gamma_ij >= constant * C * (trB_i + trB_j) / sigma_min^2 (sufficient)."""
    return bool(gamma >= constant * C * (tr_between_i + tr_between_j)
                / sigma_min_sq)


def arec(aid: str, desc: str, ok: bool, value: Any, threshold: str,
         arm: str = "", diagnosis: str = "",
         details: Optional[Dict] = None) -> Dict:
    return dict(id=aid, description=desc, arm=arm,
                status="PASS" if ok else "FAIL", value=value,
                threshold=threshold, diagnosis=diagnosis, details=details or {})


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _phi(z: float) -> float:
    """Standard normal CDF, tail-accurate: Phi(z) = 0.5*erfc(-z/sqrt(2))."""
    return float(0.5 * erfc(-z / np.sqrt(2.0)))


# =============================================================================
# Shared Gaussian block construction (EX2/EX4/EX5)
# =============================================================================


def _simplex_centroids(gamma: float, T: int, p: int) -> np.ndarray:
    """T = p_s + 1 regular-simplex vertices in R^p with pairwise EUCLIDEAN
    squared distance gamma (embedded in the first T-1 ... actually first T
    coordinates' 1-D-complement subspace; deviations go in coords T..p-1).
    Construction: u_t = (e_t - (1/T) 1)/sqrt(1 - 1/T) in R^T, zeros in R^p;
    pairwise ||u_i-u_j||^2 = 2T/(T-1); scaled by sqrt(gamma (T-1)/(2T))."""
    u = np.eye(T) - np.ones((T, T)) / T
    u /= np.sqrt(1.0 - 1.0 / T)
    w = u * np.sqrt(gamma * (T - 1.0) / (2.0 * T))
    out = np.zeros((T, p))
    out[:, :T] = w
    return out


def _spdw(p: int, rng: np.random.Generator, cond_target: float = 50.0) -> np.ndarray:
    """Random shared SPD covariance with eigenvalues log-spaced in
    [1, cond_target] and a random orthogonaleigenbasis (well-conditioned by
    design so all downstream solves sit far from the float64 floor)."""
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    ev = np.exp(np.linspace(0.0, np.log(cond_target), p))
    A = Q @ np.diag(ev) @ Q.T
    return 0.5 * (A + A.T)


def _pooled_sample_cov(X: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    mu = np.stack([X[y == k].mean(axis=0) for k in range(n_classes)])
    Xc = X - mu[y]
    return (Xc.T @ Xc) / max(len(y) - n_classes, 1)


def _crn_block(centroids: np.ndarray, devs: np.ndarray, task_of_class: np.ndarray,
               chol: np.ndarray, Z: np.ndarray, n_per_class: int):
    """Class-balanced CRN test block: X = mu_y + Z L'. Returns (X, y)."""
    n_classes = centroids.shape[0]
    means = centroids + devs
    y = np.repeat(np.arange(n_classes, dtype=np.int64), n_per_class)
    X = means[y] + Z @ chol.T
    return X, y


def _generative_logits(X: np.ndarray, means: np.ndarray, cov: np.ndarray,
                       ) -> np.ndarray:
    """q_r(x) = x' Sigma^{-1} mu_r - 0.5 mu_r' Sigma^{-1} mu_r (the argmax-
    equivalent part of the joint LDA / Bayes head, uniform priors)."""
    si = np.linalg.inv(cov)
    W = means @ si                     # (N, p)
    b = -0.5 * np.einsum("ri,ij,rj->r", means, si, means)
    return X @ W.T + b, W, b


def _calibrate_per_task(logits: np.ndarray, y_task: np.ndarray, T: int,
                        n_classes_per_task: int) -> np.ndarray:
    """Softmax-calibrated CF head: s_r = q_r - logsumexp over r' in task(r).
    (Within-task posterior calibration that sequential training CAN do; the
    cross-task offsets it induces are the TC mechanism under study.)"""
    out = np.empty_like(logits)
    for t in range(T):
        sl = slice(t * n_classes_per_task, (t + 1) * n_classes_per_task)
        out[:, sl.start:sl.stop] = (logits[:, sl.start:sl.stop]
                                    - logsumexp(logits[:, sl.start:sl.stop],
                                                axis=1, keepdims=True))
    return out


def _heads_gap_mc(X: np.ndarray, y: np.ndarray, task_of_class: np.ndarray,
                  means: np.ndarray, cov: np.ndarray, n_per_task: int
                  ) -> Dict[str, Any]:
    """MC zero-one errors (CRN: same X) of:
    * joint head: global generative LDA / Bayes (uniform priors);
    * CF arm 1 (in-scope, ass:ncm-head): per-task generative prototype heads
      with fixed within-task-constant biases (b_r = 0) -- the SLDA reading;
      identical argmax to the joint head, gap exactly 0 (feasibility limit,
      verified on the sample rather than assumed);
    * CF arm 2 (labeled extension): per-task softmax-calibrated head
      (x-dependent bias; OUTSIDE the formal head class, Remark 4.5) -- the
      CIL-realistic sequential head whose gap the bound is checked against.
    """
    logits, _, _ = _generative_logits(X, means, cov)
    T = int(task_of_class[-1]) + 1
    err_joint = float(np.mean(np.argmax(logits, axis=1) != y))
    # arm 1: literal per-task assembly with constant biases
    s_gen = np.empty_like(logits)
    for t in range(T):
        sl = slice(t * n_per_task, (t + 1) * n_per_task)
        s_gen[:, sl] = logits[:, sl]          # b_r = const within task
    pred_gen = np.argmax(s_gen, axis=1)
    err_gen = float(np.mean(pred_gen != y))
    identical = bool(np.array_equal(pred_gen, np.argmax(logits, axis=1)))
    # arm 2
    cal = _calibrate_per_task(logits, task_of_class, T, n_per_task)
    err_cal = float(np.mean(np.argmax(cal, axis=1) != y))
    return dict(err_joint=err_joint, err_gen=err_gen, err_cal=err_cal,
                gap_gen=err_gen - err_joint, gap_cal=err_cal - err_joint,
                gen_identical=identical)


# =============================================================================
# EX1 -- Theorems 3.1/3.2: infeasibility / feasibility on quadratic block losses
# =============================================================================


def run_ex1(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    recs: List[Dict] = []
    T, d = cfg["ex1_T"], cfg["ex1_d"]
    worst_feas = 0.0

    def quad(Qs: List[np.ndarray], ms: List[np.ndarray]):
        Q = sum(Qs)
        h = sum((Qi @ mi for Qi, mi in zip(Qs, ms)), np.zeros(d))
        return Q, h

    def solve_min(Q: np.ndarray, h: np.ndarray) -> np.ndarray:
        return scipy.linalg.cho_solve(scipy.linalg.cho_factor(Q, lower=True), h)

    def objective(Qs, ms, w):
        return float(sum(0.5 * (w - m) @ Qi @ (w - m) for Qi, m in zip(Qs, ms)))

    gaps_inf: List[float] = []
    distinct_min = 0
    for k in range(cfg["ex1_draws"]):
        rng = new_rng(purpose=f"i03-ex1:draw{k}")
        # T within-task quadratics with distinct minimisers (ass:incompat)
        Qs, ms = [], []
        for _ in range(T):
            A = rng.standard_normal((d, d))
            Qt = A @ A.T / d + 0.5 * np.eye(d)   # SPD, O(1) scale
            mt = rng.standard_normal(d)
            Qs.append(Qt)
            ms.append(mt)
        distinct_min += int(len({np.round(m, 8).tobytes() for m in ms}) == T)
        # off-diagonal blocks: random cross-task quadratics, distinct targets
        cross_Q, cross_m = [], []
        for i in range(T):
            for j in range(i + 1, T):
                A = rng.standard_normal((d, d))
                cross_Q.append(A @ A.T / d + 0.5 * np.eye(d))
                cross_m.append(rng.standard_normal(d))
        Qd, hd = quad(Qs, ms)                    # diagonal-only system
        Qf, hf = quad(Qs + cross_Q, ms + cross_m)  # full system
        w_cf = solve_min(Qd, hd)                 # CF-optimum: diagonal argmin
        w_gl = solve_min(Qf, hf)                 # global optimum: full argmin
        gaps_inf.append(objective(Qs + cross_Q, ms + cross_m, w_cf)
                        - objective(Qs + cross_Q, ms + cross_m, w_gl))
        # (A2) feasibility arm on the same draw: drop cross terms
        w_gl_d = solve_min(*quad(Qs, ms))
        gap_feas = objective(Qs, ms, w_cf) - objective(Qs, ms, w_gl_d)
        worst_feas = gap_feas if k == 0 else max(worst_feas, gap_feas)
    gaps = np.asarray(gaps_inf)
    recs.append(arec(
        "EX1.1", "Thm 3.1 infeasibility: TC gap > 0 strictly, 200/200 draws",
        bool(np.all(gaps > 0.0) and len(gaps) == cfg["ex1_draws"]),
        f"min {gaps.min():.3e}, median {float(np.median(gaps)):.3e}, "
        f"max {gaps.max():.3e} over {len(gaps)} draws",
        "> 0 on 200/200", arm="EXACT (quadratic solves)",
        details=dict(distinct_minimisers=f"{distinct_min}/{cfg['ex1_draws']}")))
    recs.append(arec(
        "EX1.2", "Thm 3.2 feasibility: diagonal-only loss matrix -> gap = 0 "
                 "at solver precision",
        worst_feas <= 1e-14, f"max gap over draws {worst_feas:.2e}",
        "<= 1e-14", arm="EXACT (quadratic solves)",
        details=dict(note="w_CF and w_global computed via separate Cholesky "
                          "solves of the identical system; gap = 0.5*delta'Q*delta "
                          "with delta at float64 solve floor (~1e-15).")))
    status = "PASS" if all(r["status"] == "PASS" for r in recs) else "FAIL"
    return dict(suite="EX1",
                theorem="Theorems 3.1/3.2 (infeasibility / feasibility)",
                status=status, assertions=recs,
                summary=dict(gap_min=float(gaps.min()),
                            gap_median=float(np.median(gaps)),
                            gap_max=float(gaps.max()), feas_max=worst_feas),
                runtime_s=round(time.time() - t0, 1),
                config=dict(T=T, d=d, draws=cfg["ex1_draws"], reduced=False))


# =============================================================================
# EX2 -- Lemma 3.3 + Theorem 3.6: bound validity exp(-gamma_min/16)
# =============================================================================


def run_ex2(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    recs: List[Dict] = []
    T, C = cfg["ex2_T"], cfg["ex2_C"]
    p, N = cfg["ex2_p"], cfg["ex2_T"] * cfg["ex2_C"]
    gammas = list(np.linspace(cfg["ex2_gamma_lo"], cfg["ex2_gamma_hi"],
                              cfg["ex2_n_gamma"]))
    nus = list(cfg["ex2_nu"])
    n_mc = cfg["ex2_n_mc"]
    seeds = cfg["ex2_seeds"]
    n_per_class = n_mc // N
    M = 1.0  # Condition (L): zero-one loss
    prefactor = M * C * C * T * (T - 1) / (N - 1)
    rows: List[Dict] = []
    for seed in seeds:
        # per-seed shared covariance + CRN noise (fixed across configs: CRN)
        rng = new_rng(purpose=f"i03-ex2:s{seed}")
        cov = _spdw(p, rng)
        chol = np.linalg.cholesky(cov)
        Z = rng.standard_normal((N * n_per_class, p))
        Xc = Z @ chol.T
        y = np.repeat(np.arange(N, dtype=np.int64), n_per_class)
        toc = np.repeat(np.arange(T, dtype=np.int64), C)
        for gamma in gammas:
            for nu in nus:
                cents_e = _simplex_centroids(float(gamma), T, p)
                cents = cents_e @ chol.T                # Sigma-metric preserved
                rngc = new_rng(purpose=f"i03-ex2:cfg:s{seed}:g{gamma:.3f}:n{nu:.3f}")
                D = nu * np.sqrt(float(gamma)) / 4.0    # planted max dev (*-norm)
                devs = np.zeros((N, p))
                for t in range(T):
                    for r in range(C):
                        if nu > 0 and r == 0:
                            nm = D
                        elif nu > 0:
                            nm = D * rngc.uniform(0.3, 0.9)
                        else:
                            nm = 0.0
                        if nm > 0:
                            v = rngc.standard_normal(p)
                            v[:T] = 0.0                  # orthogonal to simplex span
                            v /= np.linalg.norm(v)
                            devs[t * C + r] = nm * (chol @ v)
                means = cents[toc] + devs
                # planted-label check (exact arm)
                nu_hat = max(nu_ij(means, toc, i, j, cov)
                             for i in range(T) for j in range(T) if i != j)
                X = Xc + means[y]
                if nu > 0:
                    res = _heads_gap_mc(X, y, toc, means, cov, C)
                    gap = max(res["gap_gen"], res["gap_cal"])
                    row = dict(
                        seed=seed, gamma=round(float(gamma), 3), nu=nu,
                        nu_hat=round(float(nu_hat), 5),
                        gap_gen=round(res["gap_gen"], 6),
                        gap_cal=round(res["gap_cal"], 6), gap=round(gap, 6),
                        err_gen=round(res["err_gen"], 6),
                        err_cal=round(res["err_cal"], 6),
                        err_joint=round(res["err_joint"], 6),
                        gen_identical=res["gen_identical"])
                else:
                    # nu = 0: within-task classes coincide -> the softmax-
                    # calibrated head degenerates to ties; the generative CF
                    # arm (in-scope, b_r const) IS the joint head: gap
                    # exactly 0 (feasibility limit). Labeled.
                    logits, _, _ = _generative_logits(X, means, cov)
                    e = float(np.mean(np.argmax(logits, axis=1) != y))
                    row = dict(
                        seed=seed, gamma=round(float(gamma), 3), nu=nu,
                        nu_hat=round(float(nu_hat), 5),
                        gap_gen=0.0, gap_cal=None, gap=0.0,
                        err_gen=e, err_cal=None, err_joint=e,
                        gen_identical=True,
                        note="nu=0: calibrated arm degenerate (coincident "
                             "within-task classes); in-scope arm only")
                bound = e_tc_bound(float(gamma), M=M, C=C, T=T, N=N)
                se = float(np.sqrt(max(row["err_joint"] * (1 - row["err_joint"]),
                                       1.0 / n_mc) / n_mc))
                row.update(bound=round(bound, 5), mc_se=round(se, 6),
                           norm_margin=round(float(bound - row["gap"]), 4),
                           violated=bool(row["gap"] > bound))
                rows.append(row)
    viol = [r for r in rows if r["violated"]]
    all_identical = all(r["gen_identical"] for r in rows)
    recs.append(arec(
        "EX2.1", "Thm 3.6 (A3): empirical TC gap <= M C^2 T(T-1)/(N-1) "
                 "exp(-gamma_min/16) pointwise (both CF arms: in-scope "
                 "generative, gap exactly 0; labeled softmax-calibrated "
                 "extension)",
        len(viol) == 0,
        f"{len(rows) - len(viol)}/{len(rows)} configs x seeds satisfied; "
        f"max gap {max(r['gap'] for r in rows):.4f}; min bound "
        f"{min(r['bound'] for r in rows):.3f}; generative arm gap = 0 "
        f"bit-exact on all rows: {all_identical}",
        "0 violations (96 configs x 5 seeds)", arm="EMPIRICAL (MC, CRN, n=%d)" % n_mc,
        diagnosis="" if not viol else
        "Violations listed in details; MC SE per row recorded.",
        details=dict(violations=viol[:20], n_rows=len(rows))))
    recs.append(arec(
        "EX2.2", "planted-label check: nu_hat == planted (max over pairs), "
                 "exact construction",
        all(abs(r["nu_hat"] - r["nu"]) < 1e-3 for r in rows),
        f"max |nu_hat - nu| = {max(abs(r['nu_hat'] - r['nu']) for r in rows):.2e}",
        "< 1e-3", arm="EXACT (population algebra)"))
    recs.append(arec(
        "EX2.3", "recorded (not gated): normalised validity gap*(N-1)/(M C^2 "
                 "T(T-1)) <= exp(-gamma_min/16)",
        all(r["gap"] * (N - 1) / (C * C * T * (T - 1)) <=
            np.exp(-r["gamma"] / 16.0) for r in rows),
        "normalised margins recorded per row (G07 drift note 4: the raw "
        "prefactor 90.9 makes A3 non-binding at gamma <= 48)",
        "0 violations", arm="EMPIRICAL (recorded)"))
    status = "PASS" if all(r["status"] == "PASS" for r in recs) else "FAIL"
    return dict(suite="EX2",
                theorem="Lemma 3.3 + Theorem 3.6 (bound validity, 1/16 form)",
                status=status, assertions=recs,
                summary=dict(prefactor=prefactor,
                             n_rows=len(rows),
                             max_gap=max(r["gap"] for r in rows),
                             max_gap_cal=max(r["gap_cal"] for r in rows
                                             if r["gap_cal"] is not None),
                             min_bound=min(r["bound"] for r in rows),
                             nu0_rows_note="nu=0 rows: calibrated CF head "
                             "degenerates (coincident within-task classes); "
                             "in-scope generative arm only (gap 0.0)"),
                runtime_s=round(time.time() - t0, 1),
                config=dict(T=T, C=C, p=p, n_mc=n_mc, seeds=seeds,
                            gamma_grid=[round(float(g), 2) for g in gammas],
                            nu_grid=nus, reduced=n_mc < 200_000,
                            reduction_note=("n_mc reduced 2e5 -> %d; labeled"
                                            % n_mc) if n_mc < 200_000 else ""),
                details_rows=rows)


# =============================================================================
# EX3 -- Corollary 3.8: constant regimes (exact aligned algebra)
# =============================================================================


def _etec_1d(a: float, s_i: np.ndarray, s_j: np.ndarray,
             calibrated: bool) -> float:
    """EXACT cross-task confusion mass E_TC for the aligned 1-D family
    (centroids at -a / +a on the e1 axis, sigma = 1, class offsets s_i/s_j).

    Joint head: argmax of q_r(x) = -0.5(x-m_r)^2 over all classes (uniform
    priors) = nearest-mean. CF head: per-task softmax-calibrated scores
    q_r - logsumexp_{r' in task}. E_TC = 0.5[P(task-i x -> j) + P(j -> i)]
    computed exactly: the task assignment changes sign only at the roots of
    G(x) = max_r s_r^i(x) - max_l s_l^j(x); roots located by grid scan +
    Brent bisection to 1e-13; interval masses via tail-accurate Phi."""
    m_i = -a + np.asarray(s_i, float)
    m_j = +a + np.asarray(s_j, float)
    span = a + max(np.max(np.abs(s_i)), np.max(np.abs(s_j))) + 12.0
    xs = np.linspace(-span, span, 24001)

    def G(xv: np.ndarray) -> np.ndarray:
        qi = -0.5 * (xv[:, None] - m_i[None, :]) ** 2
        qj = -0.5 * (xv[:, None] - m_j[None, :]) ** 2
        gi = qi.max(axis=1)
        gj = qj.max(axis=1)
        if calibrated:
            gi = gi - logsumexp(qi, axis=1)
            gj = gj - logsumexp(qj, axis=1)
        return gi - gj

    g = G(xs)
    sgn = np.sign(g)
    if np.max(np.abs(g)) < 1e-12:  # degenerate (nu = 0 calibrated head)
        return float("nan")
    roots: List[float] = []
    nz = np.flatnonzero(sgn != 0)
    if len(nz) > 1:
        for k in np.flatnonzero(sgn[nz[:-1]] * sgn[nz[1:]] < 0):
            roots.append(brentq(lambda t: float(G(np.array([t]))[0]),
                                xs[nz[k]], xs[nz[k + 1]],
                                xtol=1e-13, rtol=1e-15))
    for z in xs[sgn == 0]:  # exact grid zeros (e.g. x = 0 at nu = 0)
        if all(abs(float(z) - r) > 1e-9 for r in roots):
            roots.append(float(z))
    roots.sort()
    Ci, Cj = len(m_i), len(m_j)
    if not roots:  # one task's scores dominate everywhere: complete confusion
        return 0.5
    edges = np.array([-np.inf] + roots + [np.inf])
    mids = 0.5 * (edges[:-1] + edges[1:])
    mids[0] = edges[1] - 1.0
    mids[-1] = edges[-2] + 1.0
    assign_j = G(mids) < 0
    # E_TC = 0.5 * [P(task-i sample assigned to j) + P(task-j -> i)]
    e_i = 0.0
    e_j = 0.0
    for idx in range(len(edges) - 1):
        lo, hi = edges[idx], edges[idx + 1]
        if assign_j[idx]:
            for m in m_i:
                e_i += _interval_prob(lo, hi, m) / Ci
        else:
            for m in m_j:
                e_j += _interval_prob(lo, hi, m) / Cj
    return float(0.5 * (e_i + e_j))


def _interval_prob(lo: float, hi: float, m: float) -> float:
    """P(x in (lo, hi]) for x ~ N(m, 1), tail-accurate."""
    if np.isneginf(lo):
        return _phi(hi - m)
    if np.isposinf(hi):
        return 1.0 - _phi(lo - m)
    return _phi(hi - m) - _phi(lo - m)


def run_ex3(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    recs: List[Dict] = []
    gammas = np.linspace(cfg["ex3_gamma_lo"], cfg["ex3_gamma_hi"],
                         cfg["ex3_n_gamma"])
    nus = list(cfg["ex3_nu"])
    C = cfg["ex3_C"]
    # Deterministic two-point aligned pattern: C/2 classes at +s_max, C/2 at
    # -s_max on the centroid axis (mean-zero -> centroid pinned; max deviation
    # = s_max -> planted nu = 4 s_max/sqrt(gamma) exact). All sibling classes
    # share the dominant rate exactly, reproducing G07's pilot slopes
    # (-0.130 / -0.075 / -0.067 / -0.055); a jittered full-spread pattern
    # adds sibling-class transient contamination up to 0.014 at nu=0.586 over
    # [16,256] (drift note). No sampling axes -> single deterministic run.
    def pattern(nu: float, g: float) -> Tuple[np.ndarray, np.ndarray]:
        smax = nu * np.sqrt(g) / 4.0
        s = np.where(np.arange(C) < C // 2, smax, -smax)
        return s, -s

    slope_rows: List[Dict] = []
    for nu in nus:
        log_ej, log_ec, es = [], [], []
        for g in gammas:
            s, neg_s = pattern(nu, float(g))
            ej = _etec_1d(np.sqrt(g) / 2.0, s, neg_s, calibrated=False)
            ec = (_etec_1d(np.sqrt(g) / 2.0, s, neg_s, calibrated=True)
                  if nu > 0 else float("nan"))
            es.append(ej)
            log_ej.append(np.log(ej))
            log_ec.append(np.log(ec) if np.isfinite(ec) else np.nan)
        gj = np.asarray(gammas)
        lj = np.asarray(log_ej)
        slope_j = float(np.polyfit(gj, lj, 1)[0])
        lc = np.asarray(log_ec)
        slope_c = (float(np.polyfit(gj[~np.isnan(lc)], lc[~np.isnan(lc)], 1)[0])
                   if np.isfinite(lc).sum() >= 2 else float("nan"))
        slope_rows.append(dict(
            nu=nu, slope_joint=round(slope_j, 5),
            slope_cf=round(slope_c, 5) if np.isfinite(slope_c) else None,
            exact_aligned=round(-((1.0 - nu / 2.0) ** 2) / 8.0, 5),
            interpolating=round(-(1.0 - nu) / 8.0, 5),
            e_joint_first=es[0], e_joint_last=es[-1],
            e_cal_last=(round(float(np.exp(lc[-1])), 4)
                        if np.isfinite(lc[-1]) else None)))

    def get(nu: float, key: str) -> float:
        return float(next(r[key] for r in slope_rows if r["nu"] == nu))

    # A4: nu = 0 -> slope -1/8 +- 0.01 (1/8 TIGHT)
    m0 = get(0.0, "slope_joint")
    recs.append(arec(
        "EX3.1", "(A4) nu=0: slope(log E_TC vs gamma) = -1/8 +- 0.01 "
                 "(Bhattacharyya 1/8 tight; calibrated CF arm degenerates at "
                 "nu=0, generative/joint arm gated, labeled)",
        abs(m0 - (-0.125)) <= 0.010,
        f"slope {m0:.4f} over gamma in [16,256] (deterministic exact "
        "algebra; the ~0.0045 offset is the Gaussian-tail log-prefactor, "
        "below the 0.005 G07 budget)",
        "-0.125 +- 0.010", arm="EXACT (erfc algebra)"))
    # A5: nu = 2 - sqrt(2) -> slope -1/16 +- 0.01 (1/16 TIGHT)
    nu_t = 2.0 - np.sqrt(2.0)
    mt = get(nu_t, "slope_joint")
    recs.append(arec(
        "EX3.2", "(A5) measured-nu = 2-sqrt(2) = 0.5858: slope = -1/16 "
                 "+- 0.01 (1/16 TIGHT: exact-aligned -(1-nu/2)^2/8 = 1/16)",
        abs(mt - (-0.0625)) <= 0.010,
        f"slope {mt:.4f}; exact-aligned -0.0625 (offset {mt + 0.0625:+.4f} "
        "is the tail log-prefactor)",
        "-0.0625 +- 0.010", arm="EXACT (erfc algebra)"))
    mc = get(nu_t, "slope_cf")
    e_cal = next(r["e_cal_last"] for r in slope_rows if r["nu"] == nu_t)
    recs.append(arec(
        "EX3.3", "secondary (recorded, not gated): per-task softmax-"
                 "calibrated CF head",
        True,
        f"slope {mc:.4f} at nu=2-sqrt(2); E_TC(calibrated) -> {e_cal} at "
        "gamma=256 (non-decaying near-complete cross-task confusion)",
        "recorded (finding)", arm="EXACT (recorded)",
        details=dict(
            finding="Per-task softmax calibration of the generative head "
                    "induces cross-task confusion that does NOT decay with "
                    "the margin (E_TC -> 1; slope ~ +1e-4): the calibration "
                    "discounts the true task (its mass spreads over several "
                    "plausible classes) more than the far task. The "
                    "calibrated head has an x-dependent bias, outside the "
                    "paper's prototype-head class (ass:ncm-head) -- "
                    "consistent with the paper scoping softmax-CE slope "
                    "experiments as an empirical extension (Remark 4.5).")))
    # A6: interpolating-form gate as written (honest; see drift note 1)
    bad = [(r["nu"], r["slope_joint"]) for r in slope_rows
           if r["slope_joint"] < r["interpolating"] - 0.005]
    recs.append(arec(
        "EX3.4", "(A6) every planted nu: slope >= -(1-nu)/8 - 0.005 "
                 "(G07 wording, kept as stated)",
        len(bad) == 0,
        f"{len(bad)}/{len(slope_rows)} nu-rows violate; violating nus "
        f"{sorted(set(round(b[0], 3) for b in bad))}",
        "0 violations", arm="EXACT",
        diagnosis="" if not bad else
        "G07 drift note 1: aligned-geometry exact exponent -(1-nu/2)^2/8 is "
        "steeper than the interpolating -(1-nu)/8 by nu^2/32, which exceeds "
        "the 0.005 tolerance for nu > 0.4 (G07's own pilots imply this). The "
        "interpolating form is still a valid UPPER bound pointwise -- see "
        "EX3.5, the corrected ratio-validity assertion.",
        details=dict(violating_nus=sorted(set(round(b[0], 3) for b in bad)))))
    # A6b: corrected pointwise ratio validity (consecutive-grid ratios, same
    # two-point patterns as the slope arm)
    ok6b, worst6b = True, 0.0
    for nu in nus:
        es = [_etec_1d(np.sqrt(g) / 2.0, *pattern(nu, float(g)),
                       calibrated=False) for g in gammas]
        for k in range(len(gammas) - 1):
            dg = gammas[k + 1] - gammas[k]
            ratio = es[k + 1] / es[k]
            worst6b = max(worst6b, ratio / np.exp(-(1.0 - nu) / 8.0 * dg))
            ok6b &= ratio <= 1.02 * np.exp(-(1.0 - nu) / 8.0 * dg)
    recs.append(arec(
        "EX3.5", "(A6b, corrected) pointwise ratio validity: "
                 "E_TC(g2)/E_TC(g1) <= 1.02 exp(-(1-nu)(g2-g1)/8) on the grid "
                 "for every planted nu (interpolating form is a valid bound)",
        ok6b, f"worst ratio/exp-bound {worst6b:.4f} (2 pct slack for the "
              "tail prefactor; algebra exact)", "all consecutive-grid pairs",
        arm="EXACT"))
    # A7: Table-4 boundary rows
    rows_a7 = {}
    for nu, paper_exp in ((0.51, 1.0 / 16.0), (0.73, 0.034)):
        m = get(nu, "slope_joint")
        exact = (1.0 - nu / 2.0) ** 2 / 8.0
        cons = (exact - paper_exp) / paper_exp
        rows_a7[f"nu={nu}"] = dict(slope=round(m, 5),
                                   exact_aligned=round(-exact, 5),
                                   paper_reading=round(paper_exp, 5),
                                   conservativeness=round(cons, 3))
    ok7 = (abs(rows_a7["nu=0.51"]["slope"] + 0.06938) <= 0.012 and
           abs(rows_a7["nu=0.73"]["slope"] + 0.0504) <= 0.012)
    recs.append(arec(
        "EX3.6", "(A7) boundary rows: slopes within 0.012 of exact-aligned; "
                 "paper's ~1/16 and ~0.034 readings conservative by 11 pct / "
                 "48 pct",
        ok7, rows_a7, "|slope - exact-aligned| <= 0.012", arm="EXACT"))
    # A8: doubling-margin bracket as written + corrected exponent bracket
    ok8, ok8b, ratio_rows = True, True, []
    c_locs = []
    for nu in nus:
        for g0 in cfg["ex3_doubling_gammas"]:
            e1 = _etec_1d(np.sqrt(g0) / 2.0, *pattern(nu, float(g0)),
                          calibrated=False)
            e2 = _etec_1d(np.sqrt(2.0 * g0) / 2.0, *pattern(nu, 2.0 * g0),
                          calibrated=False)
            ratio = e2 / e1
            lo, hi = np.exp(-g0 / 8.0) / 1.2, 1.2 * np.exp(-g0 / 16.0)
            c_loc = -np.log(ratio) / g0
            row = dict(nu=nu, gamma=g0, ratio=round(ratio, 6),
                       bracket=(round(lo, 6), round(hi, 6)),
                       c_loc=round(float(c_loc), 4))
            if nu <= nu_t + 1e-9:
                ok8 &= (lo <= ratio <= hi)
                ok8b &= (1.0 / 16.0 * 0.85 <= c_loc <= 1.0 / 8.0 * 1.15)
                ratio_rows.append(row)
                c_locs.append(c_loc)
            else:
                row["note"] = ("boundary nu: exact exponent below 1/16 "
                               "(interpolating-form regime), recorded")
                ratio_rows.append(row)
    recs.append(arec(
        "EX3.7", "(A8) doubling-margin bracket as written: E_TC(2g)/E_TC(g) "
                 "in [e^{-g/8}/1.2, 1.2 e^{-g/16}] (nu <= 2-sqrt(2))",
        ok8, f"{len(ratio_rows)} doubling pairs; see EX3.8 for the "
             "corrected exponent bracket", "within bracket", arm="EXACT",
        diagnosis="" if ok8 else
        "The Gaussian-tail prefactor contributes a factor ~2^{-1/4} = 0.84 "
        "per doubling (Phi(-sqrt(2)t)/Phi(-t) ~ (1/sqrt(2)) e^{-t^2/2}...), "
        "which exceeds the 1.2 lower slack at moderate/large gamma. The "
        "bracket's intended content -- WHICH EXPONENT is operative -- is "
        "gated cleanly by the local-exponent form EX3.8.",
        details=dict(rows=[r for r in ratio_rows if "note" not in r])))
    recs.append(arec(
        "EX3.8", "(A8b, corrected) local doubling exponent c_loc = "
                 "-log(E_TC(2g)/E_TC(g))/g in [1/16 x 0.85, 1/8 x 1.15] for "
                 "nu <= 2-sqrt(2)",
        ok8b,
        f"c_loc range [{min(c_locs):.4f}, {max(c_locs):.4f}] over "
        f"{len(c_locs)} pairs; boundary-nu rows recorded (c_loc < 1/16, the "
        "Cor 3.8 interpolating regime)",
        "[0.0531, 0.1438]", arm="EXACT",
        details=dict(boundary_rows=[r for r in ratio_rows if "note" in r])))
    status = "PASS" if all(r["status"] == "PASS" for r in recs) else "FAIL"
    return dict(suite="EX3",
                theorem="Corollary 3.8 (constant regimes; 1/8 and 1/16 tight; "
                        "boundary rows)",
                status=status, assertions=recs,
                summary=dict(slope_rows=slope_rows,
                             conservativeness=rows_a7),
                runtime_s=round(time.time() - t0, 1),
                config=dict(gamma_grid=[round(float(g), 1) for g in gammas],
                            nu_grid=nus, C=C, seeds="deterministic "
                            "(two-point pattern; no sampling axes)",
                            reduced=False))


# =============================================================================
# EX4 -- Thm 3.2 limit: the 1e-10 LDA check (shared statistics, CRN)
# =============================================================================


def run_ex4(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    recs: List[Dict] = []
    T, C = cfg["ex4_T"], cfg["ex4_C"]
    p, N = cfg["ex4_p"], T * C
    gammas = cfg["ex4_gammas"]
    n_stats = cfg["ex4_n_stats"]
    n_eval = cfg["ex4_n_eval"]
    seeds = cfg["ex4_seeds"]
    toc = np.repeat(np.arange(T, dtype=np.int64), C)
    ident_rows: List[Dict] = []
    for gamma in gammas:
        for seed in seeds:
            rng = new_rng(purpose=f"i03-ex4:g{gamma}:s{seed}")
            cov = _spdw(p, rng)
            chol = np.linalg.cholesky(cov)
            cents_e = _simplex_centroids(float(gamma), T, p)
            cents = cents_e @ chol.T
            nu0 = 0.23
            D = nu0 * np.sqrt(float(gamma)) / 4.0
            devs = np.zeros((N, p))
            for t in range(T):
                for r in range(C):
                    nm = D * (1.0 if r == 0 else rng.uniform(0.3, 0.9))
                    if nm > 0:
                        v = rng.standard_normal(p)
                        v[:T] = 0.0
                        v /= np.linalg.norm(v)
                        devs[t * C + r] = nm * (chol @ v)
            means = cents[toc] + devs
            # estimation sample (shared statistics for BOTH heads)
            n_pc = n_stats
            Xtr = np.vstack([means[k] + rng.standard_normal((n_pc, p)) @ chol.T
                             for k in range(N)])
            ytr = np.repeat(np.arange(N, dtype=np.int64), n_pc)
            mu_hat = np.stack([Xtr[ytr == k].mean(axis=0) for k in range(N)])
            Sig_hat = _pooled_sample_cov(Xtr, ytr, N)
            # CRN eval block
            n_ev_pc = n_eval // N
            Xev = np.vstack([means[k] + rng.standard_normal((n_ev_pc, p)) @ chol.T
                             for k in range(N)])
            yev = np.repeat(np.arange(N, dtype=np.int64), n_ev_pc)
            # CF head: per-task generative LDA heads, each with the SHARED
            # pooled Sigma_hat and within-task uniform priors log(1/C); joint
            # head: all-N LDA with global uniform priors log(1/N). Under
            # shared statistics these are the same function: the per-task
            # prior constants differ from the joint ones only by the common
            # shift log(N/C), which cannot change the argmax. Evaluated
            # literally as two separately-assembled heads on CRN samples:
            si = np.linalg.inv(Sig_hat)
            logits, _, _ = _generative_logits(Xev, mu_hat, Sig_hat)
            s_cf = np.empty_like(logits)
            s_joint = np.empty_like(logits)
            for t in range(T):  # per-task assembly (CF)
                sl = slice(t * C, (t + 1) * C)
                s_cf[:, sl] = logits[:, sl] + np.log(1.0 / C)
            s_joint[:] = logits + np.log(1.0 / N)
            pred_cf = np.argmax(s_cf, axis=1)
            pred_joint2 = np.argmax(s_joint, axis=1)
            err_cf = float(np.mean(pred_cf != yev))
            err_j = float(np.mean(pred_joint2 != yev))
            gap = err_cf - err_j
            ident_rows.append(dict(
                gamma=float(gamma), seed=seed,
                preds_identical=bool(np.array_equal(pred_cf, pred_joint2)),
                gap=gap, err=err_j,
                common_shift=round(float(np.log(N / C)), 6),
                note="heads are the same function: per-task prior constant "
                     "log(1/C) equals the joint log(1/N) up to the common "
                     "shift log(N/C), which cannot change argmax"))
    all_identical = all(r["preds_identical"] for r in ident_rows)
    max_gap = max(abs(r["gap"]) for r in ident_rows)
    recs.append(arec(
        "EX4.1", "(A9) shared-statistics LDA check: |TC gap| < 1e-10 "
                 "(pilot: exactly 0.0 -- identical heads; solver-limited)",
        all_identical and max_gap < 1e-10,
        f"gap exactly {max_gap:.1f} on all {len(ident_rows)} (gamma, seed) "
        f"configs; predictions bit-identical: {all_identical}",
        "< 1e-10", arm="EXACT (bit-identity; CRN eval)",
        details=dict(note="measured floor is float64 roundoff (~1e-16 scale); "
                          "the check is solver-limited, precisely the paper's "
                          "'attaining the feasibility limit' reading",
                     gammas=gammas)))
    # A10: independent estimation halves -> O(n^{-1/2}) gap
    g_a10 = cfg["ex4_a10_gamma"]
    n_sweep = cfg["ex4_a10_n"]
    a10_rows = []
    for n_pc in n_sweep:
        gaps = []
        dws = []
        for seed in seeds:
            rng = new_rng(purpose=f"i03-ex4:a10:n{n_pc}:s{seed}")
            cov = _spdw(p, rng)
            chol = np.linalg.cholesky(cov)
            cents = _simplex_centroids(g_a10, T, p) @ chol.T
            D = 0.23 * np.sqrt(g_a10) / 4.0
            devs = np.zeros((N, p))
            for t in range(T):
                for r in range(C):
                    nm = D * (1.0 if r == 0 else rng.uniform(0.3, 0.9))
                    if nm > 0:
                        v = rng.standard_normal(p)
                        v[:T] = 0.0
                        v /= np.linalg.norm(v)
                        devs[t * C + r] = nm * (chol @ v)
            means = cents[toc] + devs
            n_ev_pc = max(n_eval // N, 200)
            Xev = np.vstack([means[k] + rng.standard_normal((n_ev_pc, p)) @ chol.T
                             for k in range(N)])
            yev = np.repeat(np.arange(N, dtype=np.int64), n_ev_pc)
            # half A -> CF head stats; half B -> joint head stats (independent)
            XA = np.vstack([means[k] + rng.standard_normal((n_pc, p)) @ chol.T
                            for k in range(N)])
            yA = np.repeat(np.arange(N, dtype=np.int64), n_pc)
            XB = np.vstack([means[k] + rng.standard_normal((n_pc, p)) @ chol.T
                            for k in range(N)])
            yB = yA.copy()
            muA = np.stack([XA[yA == k].mean(axis=0) for k in range(N)])
            SigA = _pooled_sample_cov(XA, yA, N)
            muB = np.stack([XB[yB == k].mean(axis=0) for k in range(N)])
            SigB = _pooled_sample_cov(XB, yB, N)
            lg_cf, WA, bA = _generative_logits(Xev, muA, SigA)
            lg_j, WJ, bJ = _generative_logits(Xev, muB, SigB)
            gaps.append(abs(float(np.mean(np.argmax(lg_cf, 1) != yev))
                            - float(np.mean(np.argmax(lg_j, 1) != yev))))
            dws.append(float(np.abs(WA - WJ).max()))
        a10_rows.append(dict(n_per_class=n_pc, mean_gap=float(np.mean(gaps)),
                             sd_gap=float(np.std(gaps, ddof=1)),
                             mean_dW=float(np.mean(dws)),
                             sd_dW=float(np.std(dws, ddof=1)),
                             gaps=[round(g, 5) for g in gaps],
                             dWs=[round(w, 5) for w in dws]))
    x = np.log([r["n_per_class"] for r in a10_rows])
    # O(n^{-1/2}) holds in HEAD space: ||W_A - W_B||_inf ~ n^{-1/2}. The
    # ERROR gap is second-order flat: at the (shared-statistics) Bayes head
    # the error is stationary, so first-order n^{-1/2} head noise cancels
    # and |err_A - err_B| sits at the MC difference floor (~6e-4 here),
    # NOT at n^{-1/2}. G07's A10 wording conflated the two; recorded.
    yw = np.log([r["mean_dW"] for r in a10_rows])
    slope_a10 = float(np.polyfit(x, yw, 1)[0])
    ye = np.log([max(r["mean_gap"], 1e-6) for r in a10_rows])
    slope_err = float(np.polyfit(x, ye, 1)[0])
    recs.append(arec(
        "EX4.2", "(A10) independent-estimation arm: head difference "
                 "||W_A - W_B||_inf decays ~ n^{-1/2} (the estimation-noise "
                 "regime printed alongside the bit-exact identity check)",
        -0.75 <= slope_a10 <= -0.25,
        f"head-diff log-log slope {slope_a10:.3f} over n in {n_sweep}; mean "
        "||dW||inf "
        + ", ".join(f"{r['mean_dW']:.4f}" for r in a10_rows)
        + f"; error-gap slope {slope_err:.3f} (flat, see diagnosis)",
        "slope in [-0.75, -0.25]", arm="EMPIRICAL (5 seeds, SE in details)",
        diagnosis="" if -0.75 <= slope_a10 <= -0.25 else
        "See details: head-space scaling is the correct n^{-1/2} carrier.",
        details=dict(rows=a10_rows, gamma=g_a10,
                     error_gap_note="|err_A - err_B| is second-order flat "
                     "(slope %.3f): at the Bayes head the error is "
                     "stationary, so the first-order head noise cancels; the "
                     "measured error gaps (%s) sit at the MC difference "
                     "floor (~6e-4 at n_eval=1e5). G07's A10 'gap is "
                     "O(n^{-1/2})' is true in head space, not in error "
                     "space; drift-recorded." % (
                         slope_err,
                         ", ".join(f"{r['mean_gap']:.4f}" for r in a10_rows)))))
    status = "PASS" if all(r["status"] == "PASS" for r in recs) else "FAIL"
    return dict(suite="EX4",
                theorem="Theorem 3.2 limit (LDA check: TC gap < 1e-10; "
                        "identical heads under shared statistics)",
                status=status, assertions=recs,
                summary=dict(identity_rows=ident_rows,
                             a10_rows=a10_rows, a10_slope=slope_a10),
                runtime_s=round(time.time() - t0, 1),
                config=dict(gammas=gammas, n_stats=n_stats, n_eval=n_eval,
                            seeds=seeds, a10_gamma=g_a10, a10_n=n_sweep,
                            reduced=False))


# =============================================================================
# EX5 -- Condition (S) nu-ladder incl. boundary rows; sqrt(2) slack
# =============================================================================


def run_ex5(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    recs: List[Dict] = []
    T, C = cfg["ex5_T"], cfg["ex5_C"]
    p = cfg["ex5_p"]
    ladder = list(cfg["ex5_ladder"])
    seeds = cfg["ex5_seeds"]
    n_est = cfg["ex5_n_est"]
    toc = np.repeat(np.arange(T, dtype=np.int64), C)
    N = T * C
    rows: List[Dict] = []
    def ladder_rows(C: int) -> List[Dict]:
        """Exact-arm rows for one C. Concentrated zero-sum construction: one
        class per task deviates by D along a per-task orthogonal axis, the
        rest sit at the base point. The task centroid shifts by D/C, so max
        deviation from the centroid is m = D(1-1/C) and C*trB = D^2(1-1/C):
        the concentration ratio kappa = m/sqrt(C trB) = sqrt((C-1)/C) is the
        CAUCHY-SCHWARZ MAXIMUM over zero-sum spreads (eq:within-task-bound's
        slack). The centroid shift also inflates every pairwise gamma by
        exactly 2(D/C)^2 (orthogonal per-task axes); gamma is planted so the
        INFLATED gamma satisfies 4 D(1-1/C)/sqrt(gamma) = nu exactly."""
        N_l = T * C
        toc_l = np.repeat(np.arange(T, dtype=np.int64), C)
        out: List[Dict] = []
        for nu in ladder:
            D = 1.0
            m = D * (1.0 - 1.0 / C)
            gamma_target = 16.0 * m * m / (nu * nu)      # nu = 4 m/sqrt(g)
            gamma_plant = gamma_target - 2.0 * (D / C) ** 2
            cents = _simplex_centroids(gamma_plant, T, p)  # Sigma = I
            devs = np.zeros((N_l, p))
            for t in range(T):
                v = np.zeros(p)
                v[T + t] = 1.0
                devs[t * C] = D * v
            means = cents[toc_l] + devs
            trB = [float(np.mean(np.sum(
                (means[toc_l == t] - means[toc_l == t].mean(axis=0)) ** 2,
                axis=1))) for t in range(T)]
            nu_hat_max = max(nu_ij(means, toc_l, i, j, np.eye(p))
                             for i in range(T) for j in range(T) if i != j)
            gamma_hat_min = gamma_min_pop(means, None, toc_l)
            measured_ok = nu_hat_max <= 0.5
            crit32 = all(condition_s_trace(
                gamma_ij(means[toc_l == i].mean(axis=0),
                         means[toc_l == j].mean(axis=0), np.eye(p)),
                trB[i], trB[j], C, 1.0, 32.0)
                for i in range(T) for j in range(T) if i != j)
            crit16 = all(condition_s_trace(
                gamma_ij(means[toc_l == i].mean(axis=0),
                         means[toc_l == j].mean(axis=0), np.eye(p)),
                trB[i], trB[j], C, 1.0, 16.0)
                for i in range(T) for j in range(T) if i != j)
            out.append(dict(
                nu_planted=nu, gamma_planted=round(gamma_plant, 4),
                nu_hat_exact=round(float(nu_hat_max), 6),
                gamma_hat_min=round(float(gamma_hat_min), 6),
                trB=round(trB[0], 6), D=D,
                regime="operational" if measured_ok else "boundary",
                measured_ok=measured_ok, crit32=crit32, crit16=crit16,
                agree32=bool(crit32 == measured_ok)))
        return out

    rows = ladder_rows(C)
    # auxiliary exact arm at C = cfg['ex5_C_aux']: the structural Cauchy-
    # Schwarz slack sqrt((C-1)/C) shrinks (0.949 -> 0.995), moving the
    # criterion-32 nu-boundary from 0.474 to 0.4975 -- past the 0.49 row.
    rows_aux = ladder_rows(cfg["ex5_C_aux"])
    # ---- estimated arm (finite-sample pipeline) ----
    est_rows = []
    for nu in ladder:
        for seed in seeds:
            rng = new_rng(purpose=f"i03-ex5:n{nu:.3f}:s{seed}")
            D = 1.0
            m = D * (1.0 - 1.0 / C)
            g = 16.0 * m * m / (nu * nu) - 2.0 * (D / C) ** 2
            cents = _simplex_centroids(g, T, p)
            devs = np.zeros((N, p))
            for t in range(T):
                v = np.zeros(p)
                v[T + t] = 1.0
                devs[t * C] = D * v
            means = cents[toc] + devs
            X = np.vstack([means[k] + rng.standard_normal((n_est, p))
                           for k in range(N)])
            y = np.repeat(np.arange(N, dtype=np.int64), n_est)
            mu_hat = np.stack([X[y == k].mean(axis=0) for k in range(N)])
            Sig_hat = _pooled_sample_cov(X, y, N)
            nu_e = max(nu_ij(mu_hat, toc, i, j, Sig_hat)
                       for i in range(T) for j in range(T) if i != j)
            est_rows.append(dict(nu=nu, seed=seed, nu_hat_est=round(nu_e, 4)))
    est_summary = {}
    for nu in ladder:
        vals = [r["nu_hat_est"] for r in est_rows if r["nu"] == nu]
        est_summary[f"nu={nu}"] = dict(
            mean=round(float(np.mean(vals)), 4),
            se=round(float(np.std(vals, ddof=1) / np.sqrt(len(vals))), 4),
            max_abs_dev=round(float(max(abs(v - nu) for v in vals)), 4))
    recs.append(arec(
        "EX5.1", "(A11) nu_hat from the eq:nu-def pipeline within +-0.02 of "
                 "planted (exact arm)",
        all(abs(r["nu_hat_exact"] - r["nu_planted"]) <= 0.02 for r in rows),
        f"max |nu_hat - nu| = "
        f"{max(abs(r['nu_hat_exact'] - r['nu_planted']) for r in rows):.2e}",
        "<= 0.02", arm="EXACT (population algebra)",
        details=dict(estimated_arm=est_summary,
                     note="estimated arm (n=%d/class, 5 seeds) recorded with "
                          "SE; not gated (SE-calibrated, H05 lesson)" % n_est)))
    n_oper = sum(1 for r in rows if r["regime"] == "operational")
    n_bound = len(rows) - n_oper
    ok12 = all(r["regime"] == ("operational" if r["nu_planted"] <= 0.5
                               else "boundary") for r in rows)
    recs.append(arec(
        "EX5.2", "(A12) regime classification reproduces the planted "
                 "partition (nu <= 1/2 operational; > 1/2 boundary)",
        ok12,
        f"{n_oper} operational / {n_bound} boundary of {len(rows)} planted "
        f"rows; boundary nus {[r['nu_planted'] for r in rows if r['regime'] == 'boundary']}",
        "partition exact", arm="EXACT",
        details=dict(drift="G07 says '10 of 12 operational', mirroring the "
                           "paper's real Table 4 (L3142); the planted ladder "
                           "as written has THREE rows > 1/2 (0.51, 0.586, "
                           "0.73), so the honest planted count is 9/12."),
        diagnosis=""))
    # A13: with zero-sum within-task spreads the concentration ratio is at
    # most sqrt((C-1)/C) (Cauchy-Schwarz, eq:within-task-bound), so the
    # constant-32 trace criterion's nu-boundary sits at 0.5*sqrt((C-1)/C):
    # 0.4743 at C=10 -- INSIDE the Table-4 ladder (the 0.49 row cannot
    # satisfy both criteria at C=10 under any shared-Sigma geometry; the
    # paper's real 0.49 row passes its trace check with anisotropic real
    # features). Gate: agreement on all rows outside the structural band;
    # full agreement demonstrated on the C=100 auxiliary arm.
    band_hi = 0.5
    band_lo = 0.5 * np.sqrt((C - 1) / C)
    outside = [r for r in rows if not (band_lo < r["nu_planted"] <= band_hi)]
    in_band = [r for r in rows if band_lo < r["nu_planted"] <= band_hi]
    ok13 = all(r["agree32"] for r in outside)
    in_band_disagree = all(not r["agree32"] for r in in_band)
    ok13_aux = all(r["agree32"] for r in rows_aux)
    band_lo_aux = 0.5 * np.sqrt((cfg["ex5_C_aux"] - 1) / cfg["ex5_C_aux"])
    recs.append(arec(
        "EX5.3", "(A13) criterion eq:gamma-vs-spread @ 32 agrees with "
                 "measured nu <= 1/2 outside the structural Cauchy-Schwarz "
                 f"band ({band_lo:.4f}, 0.5] (C={C}); full agreement on the "
                 f"C={cfg['ex5_C_aux']} auxiliary arm (band boundary "
                 f"{band_lo_aux:.4f} < 0.49)",
        ok13 and ok13_aux,
        f"C={C}: {sum(r['agree32'] for r in rows)}/{len(rows)} agree; in-band "
        f"rows {[r['nu_planted'] for r in in_band]} disagree as predicted "
        f"(structural: {in_band_disagree}); C={cfg['ex5_C_aux']}: "
        f"{sum(r['agree32'] for r in rows_aux)}/{len(rows_aux)} agree",
        "agreement outside band + aux full agreement", arm="EXACT",
        diagnosis="" if (ok13 and ok13_aux) else
        "Structural: for zero-sum spreads max-dev/sqrt(C trB) <= "
        "sqrt((C-1)/C), so the trace criterion is stronger than measured "
        "nu <= 1/2 by that factor; rows inside the band cannot agree at "
        "this C. The paper's Table-4 agreement (incl. its 0.49 row, gamma "
        "9.6 vs RHS 9.4) is a property of its real anisotropic features, "
        "not reproducible at C=10 shared-Sigma geometry; flagged to owner.",
        details=dict(band=(round(float(band_lo), 4), band_hi),
                     aux_rows=rows_aux)))
    row586 = next(r for r in rows if abs(r["nu_planted"] - 0.586) < 1e-9)
    ok14 = (row586["crit16"] and not row586["measured_ok"])
    recs.append(arec(
        "EX5.4", "(A14) sqrt(2)-slack demo: nu = 0.586 row passes the "
                 "constant-16 (loose) trace criterion while failing measured "
                 "nu <= 1/2",
        ok14,
        f"crit16={row586['crit16']}, measured_ok={row586['measured_ok']} "
        f"(nu_hat {row586['nu_hat_exact']}); constant-16 bounds nu only up "
        "to sqrt(2)/2 = 0.707",
        "crit16 PASS and measured FAIL", arm="EXACT",
        details=dict(rows_16_pass_measured_fail=[
            r["nu_planted"] for r in rows if r["crit16"] and not r["measured_ok"]])))
    status = "PASS" if all(r["status"] == "PASS" for r in recs) else "FAIL"
    return dict(suite="EX5",
                theorem="Condition (S) / Table 4 (nu-ladder, boundary rows, "
                        "sqrt(2) slack between constants 16 and 32)",
                status=status, assertions=recs,
                summary=dict(rows=rows, rows_aux_C100=rows_aux,
                             estimated=est_summary),
                runtime_s=round(time.time() - t0, 1),
                config=dict(ladder=ladder, T=T, C=C, p=p, n_est=n_est,
                            seeds=seeds, reduced=False))


# =============================================================================
# Runner: pinned seeds, results store, markdown summary
# =============================================================================


DEFAULT_CFG: Dict[str, Any] = dict(
    ex1_T=10, ex1_d=32, ex1_draws=200,
    ex2_T=10, ex2_C=10, ex2_p=32, ex2_gamma_lo=4.0, ex2_gamma_hi=48.0,
    ex2_n_gamma=12, ex2_nu=[0.0, 0.13, 0.23, 0.42, 0.49, 0.51, 0.73, 0.9],
    ex2_n_mc=200_000, ex2_seeds=[0, 1, 2, 3, 4],
    ex3_gamma_lo=16.0, ex3_gamma_hi=256.0, ex3_n_gamma=10,
    ex3_nu=[0.0, 0.13, 0.23, 0.42, 0.49, 0.51, 2.0 - np.sqrt(2.0), 0.73, 0.9],
    ex3_C=10, ex3_seeds=[0, 1, 2, 3, 4], ex3_doubling_gammas=[16.0, 32.0, 64.0, 128.0],
    ex4_T=10, ex4_C=10, ex4_p=32, ex4_gammas=[5.2, 12.1, 18.9, 31.4, 39.1, 51.8],
    ex4_n_stats=2000, ex4_n_eval=100_000, ex4_seeds=[0, 1, 2, 3, 4],
    ex4_a10_gamma=5.2, ex4_a10_n=[250, 1000, 4000],
    ex5_T=10, ex5_C=10, ex5_p=32,
    ex5_ladder=[0.12, 0.13, 0.15, 0.17, 0.19, 0.23, 0.26, 0.42, 0.49, 0.51,
                0.586, 0.73],
    ex5_n_est=2000, ex5_seeds=[0, 1, 2, 3, 4], ex5_C_aux=100,
)

_SUITES = {"EX1": run_ex1, "EX2": run_ex2, "EX3": run_ex3, "EX4": run_ex4,
           "EX5": run_ex5}


def _file_sha(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def write_run_json(record: Dict[str, Any], cfg: Dict[str, Any],
                   started_utc: str, out_root: Path) -> Path:
    run_dir = out_root / "runs" / "A" / _SUITE
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"A_{_SUITE}_{record['suite']}"
    seeds = sorted({s for v in cfg.values() if isinstance(v, list)
                    and all(isinstance(x, int) and not isinstance(x, bool)
                            for x in v) and v and max(v) <= 4 and min(v) >= 0
                    for s in v})
    payload = dict(
        run_id=run_id, paper="A", experiment=_SUITE, provenance="synthetic",
        created_utc=_utcnow(), executed_utc=started_utc,
        schema="G08-S2-proposal (G01 not yet reconciled; see G08 log S7.1)",
        config=cfg, seeds=seeds, metrics=record,
        env=dict(python=platform.python_version(), numpy=np.__version__,
                 scipy=__import__("scipy").__version__,
                 machine=platform.machine(), platform=platform.platform()),
        code_version=_file_sha(Path(__file__)),
        inputs={name: _file_sha(_REPO_ROOT / "core" / f"{name}.py")
                for name in ("margin", "streaming", "gaussian_model",
                             "diagnostic")},
        derived_tables=[], drift_notes=DRIFT_NOTES)
    path = run_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    index_path = out_root / "INDEX.json"
    index: Dict[str, Any] = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            index = {}
    runs = index.setdefault("runs", [])
    runs = [r for r in runs if r.get("run_id") != run_id]
    runs.append(dict(run_id=run_id, path=str(path.relative_to(out_root)),
                     provenance="synthetic", paper="A", experiment=_SUITE,
                     suite_status=record["status"],
                     created_utc=payload["created_utc"]))
    index["runs"] = runs
    index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return path


def render_markdown(records: List[Dict[str, Any]], cfg: Dict[str, Any],
                    started_utc: str) -> str:
    lines = [
        "# Synthetic margin suites EX1-EX5 (G07) -- run summary",
        "",
        f"Run: {started_utc} | module: experiments/synthetic_margin.py (I03) | "
        "provenance: synthetic | seeds pinned via conftest SEED=0 policy",
        "",
        f"Grids: EX1 200 draws; EX2 n_mc={cfg['ex2_n_mc']} (CRN, "
        f"{cfg['ex2_n_gamma']} gammas x {len(cfg['ex2_nu'])} nus x "
        f"{len(cfg['ex2_seeds'])} seeds); EX3 exact erfc algebra "
        f"({cfg['ex3_n_gamma']} gammas x {len(cfg['ex3_nu'])} nus x "
        f"{len(cfg['ex3_seeds'])} seeds); EX4 n_eval={cfg['ex4_n_eval']}, "
        f"A10 n in {cfg['ex4_a10_n']}; EX5 12-row ladder, estimated arm "
        f"n={cfg['ex5_n_est']}/class.",
        "",
    ]
    for rec in records:
        lines.append(f"## {rec['suite']} -- {rec['theorem']}: "
                     f"**{rec['status']}** ({rec['runtime_s']}s)")
        lines.append("")
        lines.append("| id | arm | status | value | threshold |")
        lines.append("|----|-----|--------|-------|-----------|")
        for a in rec["assertions"]:
            val = str(a["value"])
            if len(val) > 130:
                val = val[:127] + "..."
            lines.append(f"| {a['id']} | {a['arm']} | {a['status']} | "
                         f"{val} | {a['threshold']} |")
        lines.append("")
        for a in rec["assertions"]:
            if a["status"] == "FAIL" and a["diagnosis"]:
                lines.append(f"- **{a['id']} diagnosis**: {a['diagnosis']}")
        lines.append("")
    lines.append("## Drift notes")
    lines.append("")
    for note in DRIFT_NOTES:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    pr = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    pr.add_argument("--suites", default="EX1,EX2,EX3,EX4,EX5")
    pr.add_argument("--selfcheck", action="store_true")
    pr.add_argument("--out-root", default=str(_REPO_ROOT / "results"))
    args = pr.parse_args(argv)

    if args.selfcheck:
        return _selfcheck()

    cfg = dict(DEFAULT_CFG)
    started = _utcnow()
    t_all = time.time()
    records = []
    for name in [s.strip().upper() for s in args.suites.split(",") if s.strip()]:
        if name not in _SUITES:
            print(f"unknown suite {name}; skipping", file=sys.stderr)
            continue
        print(f"[i03] running suite {name} ...", flush=True)
        rec = _SUITES[name](cfg)
        records.append(rec)
        path = write_run_json(rec, cfg, started, Path(args.out_root))
        print(f"[i03] {name}: {rec['status']} in {rec['runtime_s']}s -> {path}",
              flush=True)
    md = render_markdown(records, cfg, started)
    print()
    print(md)
    print(f"[i03] total wall {time.time() - t_all:.1f}s")
    return 0 if all(r["status"] == "PASS" for r in records) else 1


def _selfcheck() -> int:  # pragma: no cover - fast identity checks
    # gamma oracle cross-check (FALLBACK-I01 vs H04)
    rng = new_rng(purpose="i03-selfcheck")
    cov = _spdw(8, rng)
    a, b = rng.standard_normal((2, 8))
    if _h04_gamma_ij is not None:
        si = np.linalg.inv(cov)
        d = float(np.abs(gamma_ij(a, b, cov)
                         - _h04_gamma_ij(a, b, si)))
        assert d < 1e-9, d
        print(f"PASS gamma_ij matches H04 within {d:.2e}")
    # simplex pairwise distances
    cents = _simplex_centroids(7.5, 10, 32)
    d2 = np.sum((cents[:, None, :] - cents[None, :, :]) ** 2, axis=2)
    off = d2[~np.eye(10, dtype=bool)]
    assert np.allclose(off, 7.5, atol=1e-9), (off.min(), off.max())
    print(f"PASS simplex pairwise gamma == 7.5 (spread "
          f"{off.max() - off.min():.2e})")
    # E_TE exact closed form at nu = 0: E_TC = Phi(-sqrt(gamma)/2)
    g = 25.0
    e = _etec_1d(np.sqrt(g) / 2.0, np.zeros(10), np.zeros(10), calibrated=False)
    assert abs(e - _phi(-np.sqrt(g) / 2.0)) < 1e-12, (e, _phi(-np.sqrt(g) / 2.0))
    print(f"PASS E_TC(nu=0, joint) = Phi(-sqrt(g)/2) exactly "
          f"({e:.12f} vs {_phi(-np.sqrt(g) / 2.0):.12f})")
    # calibrated head degenerates at nu=0
    e_cal = _etec_1d(np.sqrt(g) / 2.0, np.zeros(10), np.zeros(10), calibrated=True)
    assert np.isnan(e_cal)
    print("PASS calibrated-CF arm degenerate at nu=0 (NaN), as documented")
    # I01 cross-checks (shared Sigma: pair metric == Sigma^{-1})
    if _margin.get("nu_spread_ratio") is not None:
        rng2 = new_rng(purpose="i03-selfcheck-nu")
        cov2 = _spdw(8, rng2)
        mu2 = rng2.standard_normal((6, 8))
        toc2 = np.repeat(np.arange(3, dtype=np.int64), 2)
        fn = _margin["nu_spread_ratio"]
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                dnu = abs(fn(mu2, cov2, toc2, i, j)
                          - nu_ij(mu2, toc2, i, j, cov2))
                assert dnu < 1e-9, dnu
        db = abs(_margin["e_tc_bound"](20.0, M=1.0, C=10, T=10, N=100,
                                       rate="conservative")
                 - (1.0 * 100 * 90 / 99) * np.exp(-20.0 / 16.0))
        assert db < 1e-9, db
        print("PASS I01 nu_spread_ratio == local nu_ij (shared Sigma); "
              "e_tc_bound == Thm 3.6 form")
    # EX5 algebra: criterion equivalence at a planted row
    D, nu = 1.0, 0.4
    trB = D * D * 0.9 ** 2 / 10.0          # concentrated: max dev D(1-1/C)
    g5 = 16.0 * D * D * 0.9 ** 2 / nu ** 2  # 4 D(1-1/C)/sqrt(g) = nu
    assert condition_s_trace(g5, trB, trB, 10, 1.0, 32.0)
    assert not condition_s_trace(16.0 * D * D * 0.9 ** 2 / 0.55 ** 2,
                                 trB, trB, 10, 1.0, 32.0)
    assert condition_s_trace(16.0 * D * D * 0.9 ** 2 / 0.586 ** 2,
                             trB, trB, 10, 1.0, 16.0)
    print("PASS Condition-(S) trace-criterion algebra (32 tight, 16 slack sqrt 2)")
    print(f"drift notes: {len(DRIFT_NOTES)} recorded")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
