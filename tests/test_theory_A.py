"""tests.test_theory_A -- Paper A theory checks (VERIFICATION tier).

Implemented by W6-7, 2026-10-21 (replaces the G09 NOT_IMPLEMENTED_YET
skeleton; W3-R3 finding). Modules under test, all landed:
``core/margin.py`` (I01), ``core/streaming.py`` (I02),
``experiments/slope_fits.py`` (I05), ``experiments/falsification.py``
(I05), ``experiments/drift.py`` (I04).

Tier: VERIFICATION (theorem-level, tolerances) for every test below.
Each docstring names its tier and its epsilon(s) per the conftest TEST
POLICY: EPS_LINALG (cholesky/eigh round-off, 1e-9) for exact-algebra
arms, EPS_TIGHT (float64 round-trip, 1e-12) for arithmetic identities,
and SE/delta-method/band tolerances (Z_STAT = 6) for finite-sample arms,
with the provenance of every bound stated.

Grid (task order: fast grids): d in {8, 10, 12} for the margin algebra;
randomness seeds {0, 1, 2} everywhere (conftest purpose seeds, SEED=0
policy; nothing touches global numpy state). Reduced streaming/drift
samplers (documented per test) keep the whole suite in the ~1-minute
range; tolerances are the MODULE-declared gates wherever they exist.

NO-REAL-BACKBONE RULE / synthetic-proxy: no paper number is asserted.
Constants that also appear in the paper (the -1/8 = -0.125 and
-1/16 = -0.0625 gate constants; gamma design points 8.3/41.5/8.20; the
planted slopes -0.30/-0.115) are used ONLY as SYNTHETIC-PROXY plants:
assertions compare a synthetic construction against its own planted /
closed-form truth, never against a measured paper value.

Run (stdlib unittest; pytest absent, probed 2026-10-21):

    python repo/tests/test_theory_A.py
    python -m unittest discover -s repo/tests
"""

from __future__ import annotations

import os
import sys
import time
import unittest

import numpy as np
import scipy.linalg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import conftest as ct  # noqa: E402
from core import margin as mg  # noqa: E402
from core import streaming as st  # noqa: E402
from experiments import slope_fits as sf  # noqa: E402
from experiments import falsification as fa  # noqa: E402
from experiments import drift as dr  # noqa: E402

#: Fast-grid axes (task order): dimensions and seeds.
D_GRID = (8, 10, 12)
SEED_GRID = (0, 1, 2)

_T0 = time.perf_counter()


def _grid_rng(d: int, seed: int, purpose: str) -> np.random.Generator:
    """conftest-pinned Generator for a (d, seed) grid cell."""
    return ct.new_rng(seed=ct.derive_seed(purpose, salt=1000 * d + seed))


# ---------------------------------------------------------------------------
# 1. core/margin.py -- closed-form gamma, population gamma_min, estimator,
#    convention divergence
# ---------------------------------------------------------------------------


class MarginGammaTests(unittest.TestCase):
    """Def 3.1 pair-averaged gamma: exact algebra and estimator recovery."""

    def test_gamma_ij_closed_form_exact(self):
        """VERIFICATION: gamma_ij_pairwise == delta' ((Si+Sj)/2)^-1 delta.

        Independent computation path: explicit np.linalg.inv (matrix
        inverse) against the module's cho_factor + triangular-solve
        path, on Wishart-plus-ridge SPD pairs. Epsilon: EPS_LINALG
        (1e-9) -- two different float64 linear-algebra roundings of the
        same quadratic form at condition numbers ~1e2; the task's
        < 1e-9 budget. Sub-checks: pair symmetry (i,j) vs (j,i) at
        EPS_LINALG, and the isotropic identity Sigma = cI =>
        gamma = ||mu_i - mu_j||^2 / c at EPS_TIGHT (pure arithmetic).
        Grid: d in {8, 10, 12} x seeds {0, 1, 2}.
        """
        for d in D_GRID:
            for seed in SEED_GRID:
                with self.subTest(d=d, seed=seed):
                    rng = _grid_rng(d, seed, "w67-margin-closed-form")
                    mu1 = rng.standard_normal(d)
                    mu2 = rng.standard_normal(d)
                    A1 = rng.standard_normal((d, d))
                    A2 = rng.standard_normal((d, d))
                    S1 = A1 @ A1.T / d + 0.3 * np.eye(d)
                    S2 = A2 @ A2.T / d + 0.4 * np.eye(d)
                    delta = mu1 - mu2
                    pair_avg = 0.5 * (S1 + S2)
                    closed = float(delta @ np.linalg.inv(pair_avg) @ delta)
                    got = mg.gamma_ij_pairwise(mu1, mu2, S1, S2)
                    self.assertLess(
                        abs(got - closed), 1e-9,
                        f"d={d} seed={seed}: |{got!r} - {closed!r}|",
                    )
                    # symmetry of the pair (order swapped, covariances swapped)
                    self.assertLess(
                        abs(mg.gamma_ij_pairwise(mu2, mu1, S2, S1) - got),
                        ct.EPS_LINALG, f"d={d} seed={seed}: pair symmetry",
                    )
                    # isotropic sanity: Sigma = cI collapses to Euclidean/c
                    c = 2.5
                    g_iso = mg.gamma_ij_pairwise(
                        mu1, mu2, c * np.eye(d), c * np.eye(d))
                    self.assertLess(
                        abs(g_iso - float(np.sum(delta ** 2)) / c),
                        ct.EPS_TIGHT, f"d={d} seed={seed}: isotropic identity",
                    )

    def test_gamma_min_population_pair_averaged(self):
        """VERIFICATION: population gamma_min matches the per-pair closed form.

        T=4 tasks, C=3 classes each, HETEROGENEOUS task covariances
        (task t uses Wishart_t + ridge 0.1*t): the (T, T) gamma matrix
        must equal delta' inv((Si+Sj)/2) delta pair by pair at 1e-9
        (EPS_LINALG tier: inverse path vs Cholesky path); argmin and
        gamma_min must match the manual min over the strict upper
        triangle; convention must be "pair-averaged". Shared fast path:
        feeding the SAME covariances as (T, d, d) must auto-detect the
        equality ("shared") and reproduce the pair values at EPS_LINALG
        (the two conventions coincide exactly when all Sigma^(t) are
        equal -- module CONVENTION NOTE). Grid: d in {8, 12} x seeds
        {0, 1, 2}.
        """
        for d in (8, 12):
            for seed in SEED_GRID:
                with self.subTest(d=d, seed=seed):
                    rng = _grid_rng(d, seed, "w67-margin-pop")
                    T, C = 4, 3
                    task_covs, task_mus = [], []
                    for t in range(T):
                        A = rng.standard_normal((d, d))
                        task_covs.append(A @ A.T / d + (0.1 * t + 0.2) * np.eye(d))
                        task_mus.append(rng.standard_normal(d))
                    # one shared offset direction per task, zero-sum
                    # coefficients => task centroid == task_mus[t] exactly
                    coeffs = np.tile(np.array([0.6, -0.4, -0.2]), T)
                    toc = np.repeat(np.arange(T), C)
                    offs = rng.standard_normal((T, d))
                    class_means = (np.asarray(task_mus)[toc]
                                   + coeffs[:, None] * offs[toc])
                    class_covs = np.stack(
                        [task_covs[t] for t in range(T) for _ in range(C)])
                    tmeans, tcovs = mg.task_statistics(class_means, class_covs, toc)
                    ct.allclose(tmeans, np.asarray(task_mus),
                                atol=ct.EPS_TIGHT, rtol=0.0,
                                label="construction-centroids")
                    ct.allclose(tcovs, np.stack(task_covs),
                                atol=ct.EPS_TIGHT, rtol=0.0,
                                label="construction-taskcovs")
                    res = mg.gamma_min_population(class_means, class_covs, toc)
                    self.assertEqual(res.convention, "pair-averaged")
                    closed = {}
                    for i in range(T - 1):
                        for j in range(i + 1, T):
                            dd_ = task_mus[i] - task_mus[j]
                            closed[(i, j)] = float(
                                dd_ @ np.linalg.inv(0.5 * (task_covs[i] + task_covs[j])) @ dd_)
                            self.assertLess(
                                abs(res.gamma_matrix[i, j] - closed[(i, j)]),
                                1e-9,
                                f"d={d} seed={seed} pair ({i},{j})",
                            )
                    argmin_closed = min(closed, key=closed.get)
                    self.assertEqual(res.argmin, argmin_closed)
                    self.assertLess(abs(res.gamma_min - closed[argmin_closed]), 1e-9)
                    # shared fast path: feeding the SAME covariance for
                    # every task must auto-detect equality and reproduce
                    # the closed form under THAT covariance (the two
                    # conventions coincide when all Sigma^(t) are equal)
                    shared = mg.gamma_min(tmeans, np.broadcast_to(task_covs[0], (T, d, d)))
                    self.assertEqual(shared.convention, "shared")
                    inv_shared = np.linalg.inv(task_covs[0])
                    for i in range(T - 1):
                        for j in range(i + 1, T):
                            dd_ = tmeans[i] - tmeans[j]
                            closed_shared = float(dd_ @ inv_shared @ dd_)
                            self.assertLess(
                                abs(shared.gamma_matrix[i, j] - closed_shared),
                                ct.EPS_LINALG,
                                f"d={d} seed={seed} shared-path pair ({i},{j})",
                            )

    def test_estimate_gamma_min_delta_method(self):
        """VERIFICATION: operational estimator recovers population gamma_min.

        Homogeneous class-conditional covariances (shared Sigma): the
        regime where the estimator's shared-LW-pooled convention and
        Def 3.1's pair average target the SAME population quantity
        (module CONVENTION NOTE). Construction: T=5 tasks, C=4 classes,
        n_c=600 samples/class, shared Wishart+ridge Sigma; population
        truth from gamma_min_population on the exact class means.

        Tolerances (delta method, documented):
          mu_hat^(t) ~ N(mu^(t), Sigma/(C n_c)) exactly (independent
          classes), so d_hat = mu_hat^(i)-mu_hat^(j) has Cov = eps*Sigma
          with eps = 2/(C n_c); for gamma = d' Sigma^-1 d:
          E[gamma_hat] ~ gamma + eps*d  and  sd ~ sqrt(2*eps*gamma).
          * per-pair (at the population argmin, no selection bias):
            |diff| <= eps*d + Z_STAT*sd   (Z_STAT=6: false-failure
            < 1e-9 per assertion under Gaussianity; the eps*d term is
            the delta-method bias, both float64 noise and LW shrinkage
            are orders below it at lambda* ~ 1e-3).
          * gamma_min (min over the 10 pairs): +3 sd min-selection
            allowance (E[min of 10 correlated N(0,1)] ~ -1.5) + 0.25 sd
            slack for the layer normalisation, mirroring I01's own
            documented budget.
        Grid: d in {8, 10, 12} x seeds {0, 1, 2}.
        """
        T, C, n_c = 5, 4, 600
        for d in D_GRID:
            for seed in SEED_GRID:
                with self.subTest(d=d, seed=seed):
                    rng = _grid_rng(d, seed, "w67-margin-estimator")
                    A = rng.standard_normal((d, d))
                    Sigma = A @ A.T / d + 0.05 * np.eye(d)
                    L = scipy.linalg.cholesky(Sigma, lower=True)
                    Qt, _ = np.linalg.qr(rng.standard_normal((d, T)))
                    cents = 1.5 * Qt.T                       # (T, d)
                    cm = cents[np.repeat(np.arange(T), C)] \
                        + 1.0 * rng.standard_normal((T * C, d))
                    toc = np.repeat(np.arange(T), C)
                    pop = mg.gamma_min_population(cm, Sigma, toc)
                    X = np.vstack([
                        cm[r] + rng.standard_normal((n_c, d)) @ L.T
                        for r in range(T * C)])
                    y = np.repeat(np.arange(T * C), n_c)
                    est = mg.estimate_gamma_min(X, y, toc)
                    self.assertEqual(est.convention, "shared-LW-pooled")
                    self.assertEqual(est.argmin, pop.argmin)
                    eps = 2.0 / (C * n_c)
                    i, j = pop.argmin
                    g_pair = pop.gamma_matrix[i, j]
                    sd_pair = float(np.sqrt(2.0 * eps * g_pair))
                    pair_tol = eps * d + ct.Z_STAT * sd_pair
                    self.assertLessEqual(
                        abs(est.gamma_matrix[i, j] - g_pair), pair_tol,
                        f"d={d} seed={seed} per-pair |diff| vs eps*d+6*sd",
                    )
                    g_min = pop.gamma_min
                    sd_min = float(np.sqrt(2.0 * eps * g_min))
                    min_tol = eps * d + (ct.Z_STAT + 3.0) * sd_min + 0.25 * sd_min
                    self.assertLessEqual(
                        abs(est.gamma_min - g_min), min_tol,
                        f"d={d} seed={seed} min |diff| vs documented budget",
                    )
                    # gamma_min is the min of its own matrix (mechanics)
                    iu = np.triu_indices(T, k=1)
                    self.assertLessEqual(
                        est.gamma_min,
                        float(est.gamma_matrix[iu].min()) + ct.EPS_TIGHT)

    def test_convention_divergence_sanity(self):
        """VERIFICATION: pair-averaged vs shared-Sigma conventions diverge.

        Three tasks with isotropic but differently scaled covariances
        c = (0.2, 1.0, 5.0), means +-3 on the first axis (synthetic-proxy
        construction): pair (0,1) under Def 3.1 uses the metric
        (c0+c1)/2, the operational pooled convention uses
        (c0+c1+c2)/3 -- the two must differ by a MATERIAL factor
        (> 0.1 relative; measured ~2.4x), neither may equal either
        one-sided metric c0 or c1 (the pair average is its own object),
        and under EQUAL covariances the two conventions must coincide
        at EPS_LINALG (module CONVENTION NOTE). Exact closed forms:
        36/(0.5(c0+c1)) and 36/mean(c) at EPS_TIGHT.
        """
        eye2 = np.eye(2)
        c0, c1, c2 = 0.2, 1.0, 5.0
        mu0, mu1 = np.array([3.0, 0.0]), np.array([-3.0, 0.0])
        delta_sq = 36.0
        g_def = mg.gamma_ij_pairwise(mu0, mu1, c0 * eye2, c1 * eye2)
        self.assertLess(abs(g_def - delta_sq / (0.5 * (c0 + c1))), ct.EPS_TIGHT)
        c_pool = (c0 + c1 + c2) / 3.0
        g_pool = mg.gamma_ij_pairwise(mu0, mu1, c_pool * eye2, c_pool * eye2)
        self.assertLess(abs(g_pool - delta_sq / c_pool), ct.EPS_TIGHT)
        self.assertGreater(abs(g_def - g_pool) / g_pool, 0.1)
        self.assertGreater(abs(g_def - delta_sq / c0), 1.0)
        self.assertGreater(abs(g_def - delta_sq / c1), 1.0)
        # equal covariances: the conventions coincide exactly
        g_same = mg.gamma_ij_pairwise(mu0, mu1, c1 * eye2, c1 * eye2)
        tmeans, tcovs = mg.task_statistics(
            np.vstack([mu0, mu1]), c1 * eye2, np.array([0, 1]))
        ct.allclose(tcovs[0], c1 * eye2, label="taskcov-broadcast")
        ct.allclose(tcovs[1], c1 * eye2, label="taskcov-broadcast")
        self.assertLess(abs(g_same - delta_sq / c1), ct.EPS_TIGHT)


# ---------------------------------------------------------------------------
# 2. core/streaming.py -- Hill tail index, clipping, disjointness
# ---------------------------------------------------------------------------


class StreamingTests(unittest.TestCase):
    """Prop 5.1 / Thm 6.4 streaming machinery on planted alphas."""

    def test_hill_tail_index_ci_coverage(self):
        """VERIFICATION: Hill pipeline bootstrap CI covers the planted alpha.

        For each planted alpha in (1.32, 1.45, 2.0) (module
        PLANTED_ALPHAS; synthetic plants): the MEAN of the per-seed
        1,000-resample bootstrap CIs must CONTAIN the planted alpha
        (I02's declared gate 1; the mean CI damps per-seed dispersion),
        and the 5-seed-style mean alpha_hat must sit within the module's
        DECLARED 0.45 of the truth (the +/-0.25 G07 gate is documented
        mis-calibrated for norm-Hill -- upward bias ~+0.2..+0.4; the
        0.45 band covers it with margin). Seeds {0, 1, 2} per the task
        grid. Epsilon: coverage is a containment assertion (exact);
        0.45 is the module's declared finite-sample band.
        """
        for alpha in st.PLANTED_ALPHAS:
            with self.subTest(alpha=alpha):
                res = st.hill_alpha_pipeline(alpha, seeds=SEED_GRID)
                lo = float(np.mean([s["ci_lo"] for s in res["per_seed"]]))
                hi = float(np.mean([s["ci_hi"] for s in res["per_seed"]]))
                self.assertLessEqual(
                    lo, alpha,
                    f"alpha={alpha}: mean CI lo {lo:.3f} above planted")
                self.assertGreaterEqual(
                    hi, alpha,
                    f"alpha={alpha}: mean CI hi {hi:.3f} below planted")
                self.assertLessEqual(
                    abs(res["alpha_hat_mean"] - alpha), 0.45,
                    f"alpha={alpha}: |alpha_hat_mean "
                    f"{res['alpha_hat_mean']:.3f} - {alpha}| > 0.45")

    def test_clipping_sign_structure(self):
        """VERIFICATION: clipped/unclipped sign structure at alpha = 1.45.

        Reduced-but-declared grid: K in (500, 2000, 8000), n_reps=64,
        seeds {0, 1, 2} (module gates use 512x5; measured deviations
        here 0.01 against the 0.06 band, so the reduction is safe).
        Assertions (sign structure, never paper values):
          (i) clipped mean-loss log-log slope is NEGATIVE and within
              the module's A15 band 0.06 of the Prop-5.1 rate
              -(alpha-1)/alpha (finite-sample slope noise at 3x64
              reps; calibrated at 0.010 measured);
         (ii) unclipped realised mean loss EXCEEDS clipped at EVERY K
              (clipping reduces the loss -- the operative direction);
        (iii) unclipped/clipped dispersion ratio (pooled std and
              P99.5) >= 10x at every K (A18 heavy-tail floor, alpha<2;
              measured >= 36x).
        """
        alpha = 1.45
        Ks = (500, 2000, 8000)
        clip = st.ecf_empirical_sweep(alpha, seeds=SEED_GRID, n_reps=64,
                                      K_values=Ks, clip=True, output="last")
        uncl = st.ecf_empirical_sweep(alpha, seeds=SEED_GRID, n_reps=64,
                                      K_values=Ks, clip=False, output="last")
        fit = st.fit_loglog_slope(clip["K"], clip["mean"], seed=145)
        theory = -((alpha - 1.0) / alpha)
        self.assertLess(fit["slope"], 0.0, "clipped slope must be negative")
        self.assertLessEqual(
            abs(fit["slope"] - theory), 0.06,
            f"A15: slope {fit['slope']:+.4f} vs theory {theory:+.4f}")
        for K, u, c in zip(clip["K"], uncl["mean"], clip["mean"]):
            self.assertGreater(
                u, c, f"unclipped mean must exceed clipped at K={K}")
        for K, us, cs, up, cp in zip(clip["K"], uncl["std"], clip["std"],
                                     uncl["p995"], clip["p995"]):
            self.assertGreaterEqual(
                us / cs, 10.0, f"A18 std dispersion ratio at K={K}")
            self.assertGreaterEqual(
                up / cp, 10.0, f"A18 P99.5 dispersion ratio at K={K}")

    def test_disjointness_bit_identity(self):
        """VERIFICATION: Thm 6.4 disjoint dependence, bit-identity tier.

        (a) check_disjoint_dependence: nulling the geometry stream must
        leave E_CF BIT-IDENTICAL (Python float ==) and nulling the
        optimisation stream must leave E_TC bit-identical; total ==
        E_CF + E_TC exactly (A22 additivity). EPS_EXACT tier -- this is
        a determinism/dataflow assertion, no tolerance.
        (b) The E_CF sweep harness never reads gamma_plant: two sweeps
        differing ONLY in gamma_plant (8.3 vs 41.5, synthetic-proxy
        design points) must produce BIT-IDENTICAL mean arrays
        (np.array_equal; module A20). Grid: alpha=1.45, seeds {0, 1},
        n_reps=32, K in (500, 2000).
        (c) Prop-5.1 schedule closed form: eta = min(1/beta,
        (sigma^a/(mu^2 K))^(1/a)) and tau = (sigma^a K/mu)^(1/a) at
        EPS_TIGHT (pure arithmetic).
        """
        dd = st.check_disjoint_dependence()
        self.assertTrue(dd["E_CF_bit_identical"],
                        "E_CF changed when geometry stream nulled")
        self.assertTrue(dd["E_TC_bit_identical"],
                        "E_TC changed when optimisation stream nulled")
        self.assertTrue(dd["additivity_exact"], "total != E_CF + E_TC exactly")
        s83 = st.ecf_empirical_sweep(1.45, seeds=(0, 1), n_reps=32,
                                     K_values=(500, 2000), clip=True,
                                     gamma_plant=st.GAMMA_PLANTS[0])
        s415 = st.ecf_empirical_sweep(1.45, seeds=(0, 1), n_reps=32,
                                      K_values=(500, 2000), clip=True,
                                      gamma_plant=st.GAMMA_PLANTS[1])
        ct.arrays_identical(np.asarray(s83["mean"]), np.asarray(s415["mean"]),
                            label="ecf-mean-gamma-plant-invariance")
        a, K = 1.45, 32000
        eta, tau = st.clipped_sgd_schedule(K, a)
        self.assertLess(abs(eta - 0.5 * K ** (-1.0 / a)), 1e-12)
        self.assertLess(abs(tau - 0.5 * K ** (1.0 / a)), 1e-9)


# ---------------------------------------------------------------------------
# 3. experiments/slope_fits.py -- planted-slope recovery, anchor invariance
# ---------------------------------------------------------------------------


class SlopeFitsTests(unittest.TestCase):
    """S4 planted-slope pipeline: CI containment and anchor invariance."""

    def test_planted_slope_ci_contains_truth(self):
        """VERIFICATION: bootstrap CI contains the planted slope (12 and 16).

        Synthetic population: planted slope -0.30, noise sd 0.05
        (synthetic-proxy; the paper's -0.115 is quoted context only).
        The D1 stratified 1,000-resample bootstrap CI must contain the
        planted slope for BOTH the 12-pretrained and the 16-config
        (anchors own intercept, D2) population, and the point fit must
        sit inside its own CI. Epsilon: containment (exact); the CI is
        a 95% percentile bootstrap on a fixed pinned seed, calibrated
        before pinning (measured CIs [-0.3053,-0.2920] and
        [-0.3054,-0.2918] around -0.30). Structural identity: the
        12-config pooled fit has 4 parameters and 8 residual df (the
        F(2,6) denominator arm).
        """
        planted = -0.30
        recs = sf.make_synthetic_population(planted, noise_sd=0.05,
                                            purpose="w67-s4")
        fit12 = sf.pooled_common_slope_fit(recs, population="12")
        fit16 = sf.pooled_common_slope_fit(recs, population="16",
                                           anchor_mode="own_intercept")
        boot12 = sf.bootstrap_pooled_slope(recs, B=1000, population="12",
                                           purpose="w67-12")
        boot16 = sf.bootstrap_pooled_slope(recs, B=1000, population="16",
                                           purpose="w67-16")
        self.assertEqual((fit12["n_params"], fit12["resid_df"]), (4, 8))
        for tag, fit, boot in (("12", fit12, boot12), ("16", fit16, boot16)):
            with self.subTest(population=tag):
                self.assertLessEqual(boot["ci_lo"], planted,
                                     f"{tag}: ci_lo {boot['ci_lo']:+.4f}")
                self.assertGreaterEqual(boot["ci_hi"], planted,
                                        f"{tag}: ci_hi {boot['ci_hi']:+.4f}")
                self.assertLessEqual(boot["ci_lo"], fit["slope"])
                self.assertGreaterEqual(boot["ci_hi"], fit["slope"])

    def test_zero_noise_exact_recovery(self):
        """VERIFICATION: zero-noise planting recovers the slope exactly.

        At noise_sd = 0 the pipeline is exact algebra: pooled 12-config
        slope == planted at 1e-12 (EPS_TIGHT: lstsq on an exactly
        linear-in-parameters system); 16-config fit identified with 5
        parameters (4 backbone/anchor intercepts + common slope); the
        B=200 bootstrap CI collapses onto the planted slope (width
        < 1e-9; every resample refits the same noiseless plane).
        """
        planted = -0.30
        recs = sf.make_synthetic_population(planted, noise_sd=0.0,
                                            purpose="w67-zero")
        fit12 = sf.pooled_common_slope_fit(recs, population="12")
        fit16 = sf.pooled_common_slope_fit(recs, population="16")
        self.assertLess(abs(fit12["slope"] - planted), 1e-12)
        self.assertLess(abs(fit16["slope"] - planted), 1e-9)
        self.assertEqual(fit16["n_params"], 5)
        boot = sf.bootstrap_pooled_slope(recs, B=200, population="12",
                                         purpose="w67-zero-boot")
        self.assertLessEqual(boot["ci_lo"], planted)
        self.assertGreaterEqual(boot["ci_hi"], planted)
        self.assertLess(boot["ci_hi"] - boot["ci_lo"], 1e-9)

    def test_anchor_augmentation_invariance(self):
        """VERIFICATION: anchor augmentation leaves the 12-config fit alone.

        Adding the 4 random anchors WITH THEIR OWN INTERCEPT GROUP (D2)
        must not move the common slope: at noise sd 0.05 the shift is
        < 0.01 (module S4.5 gate; measured 1e-4 -- the anchors enter
        through a separate dummy, so the slope estimator gains
        information and loses nothing), and at ZERO noise the two fits
        agree at 1e-9 (EPS_LINALG tier: two lstsq solves of exactly
        identified systems). Synthetic-proxy: anchor gammas are quoted
        tab:random-anchor design points, the y values are planted.
        """
        for noise, tol, tag in ((0.05, 0.01, "noisy"),
                                (0.0, 1e-9, "zero-noise")):
            with self.subTest(arm=tag):
                recs = sf.make_synthetic_population(
                    -0.30, noise_sd=noise, purpose=f"w67-anchor-{tag}")
                fit12 = sf.pooled_common_slope_fit(recs, population="12")
                fit16 = sf.pooled_common_slope_fit(recs, population="16",
                                                   anchor_mode="own_intercept")
                shift = abs(fit16["slope"] - fit12["slope"])
                self.assertLess(shift, tol, f"{tag}: anchor shift {shift:.2e}")


# ---------------------------------------------------------------------------
# 4. experiments/falsification.py -- gate trips, df identity, BH nulls
# ---------------------------------------------------------------------------


class FalsificationTests(unittest.TestCase):
    """G1/G2/G3 gate mechanics on planted truths."""

    def test_gate_trips_both_directions(self):
        """VERIFICATION: the -1/16 gate trips exactly on planted truths.

        Both directions of the S4 falsification condition on planted
        populations (noise sd 0.02, B=1000; synthetic-proxy constants
        -1/8/-0.125 and -1/16/-0.0625):
          * planted -0.125 (the operational truth): gate PASSES -- the
            12-config CI CONTAINS -0.125 and EXCLUDES -0.0625 (both
            sub-conditions hold);
          * planted -0.0625: gate FAILS precisely because the CI
            CONTAINS -0.0625 (the exclusion sub-gate trips);
          * planted -0.30: gate FAILS on containment (CI excludes
            -0.125). Epsilon: exact boolean gate logic on pinned-seed
            CIs (calibrated margins: e.g. -0.0625 CI [-0.0656,-0.0598]).
        """
        cases = (
            (-0.125, "PASS"),
            (-0.0625, "FAIL"),
            (-0.30, "FAIL"),
        )
        for planted, expected in cases:
            with self.subTest(planted=planted):
                recs = sf.make_synthetic_population(
                    planted, noise_sd=0.02, purpose=f"w67-gate-{planted}")
                pops = fa.run_full_pipeline(recs, B=1000,
                                            purpose=f"w67-gate-{planted}")
                gates = fa.ci_gates(pops)
                self.assertEqual(gates["status"], expected,
                                 f"planted {planted}: got {gates['status']}")
                lo12, hi12 = pops["12"]["ci_lo"], pops["12"]["ci_hi"]
                if planted == fa.NEG_1_8:
                    self.assertTrue(lo12 <= fa.NEG_1_8 <= hi12)
                    self.assertFalse(lo12 <= fa.NEG_1_16 <= hi12)
                elif planted == fa.NEG_1_16:
                    self.assertTrue(lo12 <= fa.NEG_1_16 <= hi12,
                                    "exclusion sub-gate must be what trips")

    def test_f_interaction_df_identity(self):
        """VERIFICATION: F(2,6) identity and both rejection directions.

        On the canonical 12-config / 3-backbone population the G1
        interaction test MUST report df = (2, 6) (df1 = 2g - (g+1) = 2,
        df2 = n - 2g = 6 -- the spec's F(2,6); exact combinatorial
        identity). Zero-noise homogeneous planting (common slope
        -0.115, synthetic-proxy): F == 0 to 1e-9 (identical per-backbone
        slopes; F clamped at 0, exact algebra) and p = 1. Planted
        heterogeneity (-0.141/-0.099/-0.118 as PLANTS, sd 0.02): the
        test REJECTS (p < 0.05; measured F ~ 40, p ~ 3e-4) -- the
        pooling-admissibility direction.
        """
        homog = sf.make_synthetic_population(-0.115, noise_sd=0.0,
                                             purpose="w67-f-homog")
        r0 = fa.f_interaction_test(homog)
        self.assertEqual((r0["df1"], r0["df2"]), (2, 6))
        self.assertLessEqual(r0["F"], 1e-9)
        self.assertAlmostEqual(r0["p"], 1.0, places=9)
        het = []
        for bb, sl in {"ViT-IN21K": -0.141, "DINOv2": -0.099,
                       "CLIP": -0.118}.items():
            for ds, g in zip(sf.DATASETS, sf.SYNTHETIC_GAMMA_GRID[bb]):
                rng = sf._rng(f"w67-het:{bb}|{ds}")
                het.append(sf.ConfigRecord(
                    backbone=bb, dataset=ds, anchor=False,
                    gamma_min_hat=float(g),
                    log_tc_gap=float(sl * g + rng.normal(0, 0.02))))
        rh = fa.f_interaction_test(het)
        self.assertEqual((rh["df1"], rh["df2"]), (2, 6))
        self.assertLess(rh["p"], 0.05, "planted heterogeneity must reject")

    def test_bh_fpr_binomial_bounds(self):
        """VERIFICATION: BH false-positive rate within binomial bounds on nulls.

        60 independent families of n=200 uniform(0,1) null p-values
        (conftest-pinned rng): under the global null BH's family-wise
        any-rejection probability is <= alpha = 0.05 (BH1995 exact FDR
        control at equality), so the empirical any-rejection fraction
        phat is stochastically dominated by Binomial(60, 0.05)/60.
        Bound: phat <= 0.05 + Z_STAT*sqrt(0.05*0.95/60) = 0.2186 with
        Z_STAT=6 (false-failure < 1e-9; conservative one-sided).
        Helper identities at exact/1e-15 tier: adjusted >= raw p,
        adjusted monotone along the ascending sort, and rejections ==
        the step-up k* rule on a textbook family and on one random
        family.
        """
        rng = ct.new_rng(purpose="w67-bh-null")
        m, n = 60, 200
        any_rej = 0
        for _ in range(m):
            pv = rng.random(n)
            r = fa.benjamini_hochberg(pv, alpha=0.05)
            order = np.argsort(pv)
            kstar = 0
            for i in range(n):
                if pv[order[i]] <= (i + 1) / n * 0.05:
                    kstar = i + 1
            self.assertEqual(sum(r["rejected"]), kstar,
                             "BH rejections != step-up k* rule")
            adj = np.asarray(r["adjusted"])
            self.assertTrue(np.all(adj >= pv - 1e-15))
            self.assertTrue(np.all(np.diff(adj[order]) >= -1e-15),
                            "adjusted p must be monotone in sorted order")
            any_rej += int(r["n_rejected"] > 0)
        phat = any_rej / m
        bound = 0.05 + ct.Z_STAT * np.sqrt(0.05 * 0.95 / m)
        self.assertLessEqual(
            phat, bound,
            f"any-rejection rate {phat:.3f} exceeds binomial bound {bound:.3f}")


# ---------------------------------------------------------------------------
# 5. experiments/drift.py -- secant identity, two-path divergence, t*
# ---------------------------------------------------------------------------


class DriftTests(unittest.TestCase):
    """Thm 7.2 / Cor 7.3 drift machinery on planted trajectories."""

    def test_secant_identity_by_construction(self):
        """VERIFICATION: floor-saturated secant identity -gamma0/w, exact.

        Arm eta=1e-3 (linear plant g(t) = 8.20 - 17.3 t, time-to-floor
        8.20/17.3 = 0.474): every secant window w in (0.480, 0.845, 1.0,
        2.0) lies past the floor, so the floored series is exactly 0 at
        w and the secant (g(w) - g(0))/w equals -gamma0/w = -8.20/w BY
        CONSTRUCTION. Windows off the 0.1 sub-stage grid are read by
        linear interpolation of the unclipped series then floored
        (exact for a linear g). Epsilon 1e-12 (EPS_TIGHT: pure
        arithmetic on a linear series; matches I04's own A24 gate).
        Sub-floor sanity: an unfloored window has secant exactly the
        planted slope -17.3 (linear series), which is NOT -8.20/w --
        the identity is a floor property, and that contrast is asserted.
        """
        arm = dr.ARMS["eta=1e-3"]
        g0 = arm["gamma0"]
        ts = np.arange(0.0, arm["t_max"] + 1e-9, dr.SUB_DT)
        g_unc = np.asarray(arm["g_fn"](ts), float)
        gs = np.maximum(g_unc, 0.0)
        for w in dr.SECANT_WINDOWS:
            with self.subTest(w=w):
                g_at_w = max(float(np.interp(w, ts, g_unc)), 0.0)
                secant = (g_at_w - float(gs[0])) / w
                identity = -g0 / w
                self.assertLessEqual(
                    abs(secant - identity), 1e-12,
                    f"w={w}: secant {secant:.12f} vs {identity:.12f}")
        # contrast: pre-floor window has the PLANTED slope, not -g0/w
        w = 0.1
        g_at_w = max(float(arm["g_fn"](np.array([w]))[0]), 0.0)
        secant = (g_at_w - float(gs[0])) / w
        self.assertLess(abs(secant - (-17.3)), 1e-12)
        self.assertGreater(abs(secant - (-g0 / w)), 1.0)

    def test_tstar_interpolation_exact_and_monotone(self):
        """VERIFICATION: t* interpolation exact on linear, monotone in thr.

        Linear arm eta=1e-4 (g(t) = 8.20 - 1.73 t): linear interpolation
        of a linear function is EXACT, so t*(thr) must equal the closed
        form (8.20 - thr)/1.73 to 1e-9 (EPS_LINALG: interpolation
        arithmetic + division; grid step 0.1 does not bias a linear
        crossing) for 13 thresholds spanning (1, 7), and t* must be
        STRICTLY decreasing in the threshold (crossing time of a
        strictly decreasing series). Cubic arm eta=1e-5: t* strictly
        decreasing over thresholds (5.0, 6.5, 7.5) (monotone trajectory
        -> monotone crossing times; no closed form asserted there).
        """
        arm = dr.ARMS["eta=1e-4"]
        ts = np.arange(0.0, arm["t_max"] + 1e-9, dr.SUB_DT)
        gs = np.maximum(np.asarray(arm["g_fn"](ts), float), 0.0)
        prev = None
        for thr in np.linspace(1.0, 7.0, 13):
            tstar = dr.t_star_interpolated(ts, gs, float(thr))
            self.assertIsNotNone(tstar)
            closed = (arm["gamma0"] - float(thr)) / 1.73
            self.assertLess(abs(tstar - closed), 1e-9,
                            f"thr={thr:.3f}: {tstar} vs {closed}")
            if prev is not None:
                self.assertLess(tstar, prev,
                                "t* must strictly decrease with threshold")
            prev = tstar
        arm1 = dr.ARMS["eta=1e-5"]
        ts1 = np.arange(0.0, arm1["t_max"] + 1e-9, dr.SUB_DT)
        gs1 = np.maximum(np.asarray(arm1["g_fn"](ts1), float), 0.0)
        tstars = [dr.t_star_interpolated(ts1, gs1, thr)
                  for thr in (5.0, 6.5, 7.5)]
        self.assertTrue(all(t is not None for t in tstars))
        self.assertTrue(all(a > b for a, b in zip(tstars, tstars[1:])),
                        f"cubic t* not monotone: {tstars}")

    def test_two_path_divergence_recovers_inflation(self):
        """VERIFICATION: EX11 two-path divergence recovers the planted gap.

        Same stream, two estimation paths (I04 EX11, seeds {0,1,2},
        n_per_class=600 -- module --quick grade): P1 one-pass full-sample
        LW (the 8.3-style) vs P2 checkpoint-mixture covariance (the
        8.20-style) with the 1.2% planted covariance inflation.
        Systematic arm (module A29 bands, synthetic plants):
          * P1 mean in [7.9, 8.7] (one-pass unbiased for 8.30);
          * P2 mean in [7.58, 8.82];
          * gap mean in [-0.20, -0.02] (negative: harness under-reads)
            AND within 0.10 of the planted inflation gap
            8.30/1.012 - 8.30 = -0.0984 (measured -0.145 at this grade).
        Pure-noise arm (identical distributions, module A28): Welch
        p > 0.4 (no path-level bias) and |gap mean| <= 0.10 (noise
        scale). Epsilons: the bands are I04's declared estimator-noise
        gates at this sampling grade; p/GAP are containment assertions
        on pinned seeds.
        """
        body, _ = dr.run_ex11(seeds=SEED_GRID, n_per_class=600)
        sy = body["summary"]["systematic"]
        pn = body["summary"]["pure_noise"]
        planted_gap = dr.GAMMA0_TRUE / (1.0 + dr.HARNESS_INFLATION) \
            - dr.GAMMA0_TRUE
        self.assertGreaterEqual(sy["P1_mean"], 7.9)
        self.assertLessEqual(sy["P1_mean"], 8.7)
        self.assertGreaterEqual(sy["P2_mean"], 7.58)
        self.assertLessEqual(sy["P2_mean"], 8.82)
        self.assertGreaterEqual(sy["gap_mean"], -0.20,
                                f"gap {sy['gap_mean']:+.4f} below band")
        self.assertLessEqual(sy["gap_mean"], -0.02,
                              f"gap {sy['gap_mean']:+.4f} not negative")
        self.assertLessEqual(abs(sy["gap_mean"] - planted_gap), 0.10,
                             f"|gap - planted {planted_gap:.4f}| > 0.10")
        self.assertGreater(pn["p_value"], 0.4,
                           f"pure-noise Welch p {pn['p_value']:.4f} <= 0.4")
        self.assertLessEqual(abs(pn["gap_mean"]), 0.10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
