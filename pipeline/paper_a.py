"""pipeline.paper_a -- every real-backbone number of Paper A (4828) from feature caches.

One pass per (backbone, dataset, seed) computes the class statistics, trains the two heads the
TC gap needs, and evaluates every frozen-feature method on them, so each quantity in the paper
comes from one set of trained objects.  ``run_all`` assembles the per-seed rows into the
JSON that ``pipeline.tables`` renders.  The GPU-only drift experiment is ``pipeline.drift``.

Protocol summary (paper appendices; see the function docstrings):
  gamma_min        app:gamma-computation, protocol A statistics, five class orderings (seeds 0-4)
  TC gap           app:tc-gap-computation: test CE of the diagonal-only head minus the all-blocks head
                   (both AdamW, wd 1e-4, 20 epochs, lr 1e-3, on standardized features; or L-BFGS)
  slopes           per-backbone OLS of log TC gap on gamma_min over the 4 datasets (seed means);
                   pooled common slope with backbone intercepts over the 12 configurations and over
                   the 16 with the random-ViT anchors; bootstrap CIs over orderings; interaction F(2,6)
  predictor        single-predictor R^2 of final accuracy on gamma_min, linear-probe accuracy and NCM
                   accuracy; nested Delta R^2 with F(1, n-4); leave-one-dataset-out Delta R^2
  alpha            Hill estimator on 1000 warm-up gradient norms (k = 50, 1000 bootstrap resamples)
  E_CF sweep       K in {500, 2000, 8000, 32000}, clipped (Prop 5.1 schedule) against unclipped, two
                   backbones on CIFAR-100; E_CF(K) = test CE(SGD head) - test CE(diagonal-only head)
  method           Algorithm 1 final accuracy ("ours": the diagonal head of the paper's theory trained by
                   clipped SGD with the Prop 5.1 schedule; "ours_classil": the same loop on the CE over the
                   classes seen so far, kept as the alternative reading of the pseudocode) with the
                   frozen-feature baselines (sequential head, joint head, within-task AdamW head with and
                   without the de-centering fix, SLDA, NCM, FeCAM-style, RanPAC-style)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import special, stats as sps

from pipeline import backend as _backend
from pipeline import registry as R
from pipeline.features import cache_exists, load_pair
from pipeline.heads import (FeCAMHead, RanPACHead, clipped_sgd_head, curvature_probe, de_center, hill_estimate,
                            lda_head, ncm_head, prop51_schedule, sequential_head_baseline, smoothness_bound, train_joint,
                            train_within_task, warmup_gradient_norms)
from pipeline.stats import ClassStats


def _ms(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return dict(mean=float(v.mean()) if v.size else None, sd=float(v.std(ddof=1)) if v.size > 1 else 0.0, n=int(v.size),
                values=[float(x) for x in v])


def _agg(rows: List[dict], keys: Sequence[str]) -> dict:
    return {k: _ms([r[k] for r in rows]) for k in keys}


# ---------------------------------------------------------------------------
# One (backbone, dataset, seed): statistics, the two heads, every frozen-feature method
# ---------------------------------------------------------------------------

def seed_pass(backbone: str, dataset: str, seed: int, Xtr, ytr, Xte, yte, *, optimizer="adamw", lam=1e-4,
              protocol=R.HEAD_PROTOCOL["A"], alpha_row: Optional[dict] = None, K_method=2000, K_sweep=None,
              batch=128, ranpac_M=10000, device=None, with_methods=True, schedule: str = "printed") -> dict:
    """Everything Paper A measures on one configuration and one class ordering.

    Returns a flat dict: gamma_min, pair, nu, rhs, lam_lw, tc_gap, L_dagger, L_star, acc_* (dagger, star,
    lda, ncm), and, with ``with_methods``, the method accuracies (ours, sequential_head, joint, within_task,
    within_task_fix, slda, ncm, fecam, ranpac, avg_incremental, ours_stages) and, with ``K_sweep``, the
    E_CF arms."""
    ordering = R.class_ordering(dataset, seed)
    toc = R.task_of_class(dataset, ordering); tasks = R.tasks_from_ordering(dataset, ordering)
    N = R.DATASETS[dataset].n_classes
    st = ClassStats.from_features(Xtr, ytr, toc, protocol="A")
    g, pair = st.gamma_min(); cs = st.condition_S(Xtr, ytr)
    Ztr, Zte = st.transform(Xtr), st.transform(Xte)                   # app:gamma step 1: standardized features
    tag = f"A:{backbone}:{dataset}:s{seed}"
    dag = train_within_task(Ztr, ytr, toc, lam=lam, optimizer=optimizer, protocol=protocol, seed_purpose=f"{tag}:dag", device=device)
    star = train_joint(Ztr, ytr, toc, lam=lam, optimizer=optimizer, protocol=protocol, seed_purpose=f"{tag}:star", device=device)
    L_dag, L_star = dag.ce_loss(Zte, yte), star.ce_loss(Zte, yte)
    lda, ncm = lda_head(st), ncm_head(st)
    cond = {k: v for k, v in cs.items() if k not in ("gamma_min", "pair", "rhs")}
    row = dict(seed=seed, gamma_min=g, pair=[int(pair[0]), int(pair[1])], nu=cs["nu"], rhs=cs["rhs"], lam_lw=st.lam_lw, condition_S=cond,
               fit_dagger=[dict(max_grad=b.get("max_grad"), steps=b.get("steps", b.get("nit")), backend=b.get("backend")) for b in dag.info.get("blocks", [])],
               fit_star=dict(max_grad=star.info.get("fit", {}).get("max_grad"), steps=star.info.get("fit", {}).get("steps", star.info.get("fit", {}).get("nit"))),
               gamma_min_unshrunk=(st.gamma_min_unshrunk(Xtr, ytr) if st.lam_lw < 0.05 else None),
               gamma_min_eq1=st.gamma_min_pairwise(),
               tc_gap=L_dag - L_star, L_dagger=L_dag, L_star=L_star,
               acc_dagger=dag.accuracy(Zte, yte), acc_star=star.accuracy(Zte, yte),
               acc_lda=lda.accuracy(Zte, yte), acc_ncm=ncm.accuracy(Zte, yte), tc_gap_lda=0.0)
    def _sched(K):
        beta = alpha_row["beta_bound"] if schedule == "consistent" and alpha_row.get("beta_bound") else alpha_row["beta_hat"]
        return prop51_schedule(K, alpha_row["alpha"], alpha_row["sigma_hat"], alpha_row["mu_hat"], beta, schedule=schedule)

    if with_methods and alpha_row is not None:
        eta, tau = _sched(K_method)
        ours, stages = clipped_sgd_head(Ztr, ytr, toc, tasks, K=K_method, eta=eta, tau=tau, batch=batch, lam=lam, scope="diagonal",
                                        seed_purpose=f"{tag}:ours", X_eval=Zte, y_eval=yte, device=device)
        ours_cil, _ = clipped_sgd_head(Ztr, ytr, toc, tasks, K=K_method, eta=eta, tau=tau, batch=batch, lam=lam, scope="seen",
                                       seed_purpose=f"{tag}:ours-cil", device=device)
        seq, _ = sequential_head_baseline(Ztr, ytr, toc, tasks, K=K_method, eta=eta, batch=batch, lam=lam,
                                          seed_purpose=f"{tag}:seq", device=device)
        rp = RanPACHead(Ztr.shape[1], N, M=ranpac_M, seed_purpose=f"{tag}:ranpac", device=device)
        for cls in tasks:
            m = np.isin(ytr, cls); rp.update(Ztr[m], ytr[m])
        rp.solve()
        row.update(ours=ours.accuracy(Zte, yte), ours_stages=[float(a) for a in stages], avg_incremental=float(np.mean(stages)),
                   ours_ce=ours.ce_loss(Zte, yte), ours_fix=de_center(ours, st).accuracy(Zte, yte), ours_classil=ours_cil.accuracy(Zte, yte),
                   sequential_head=seq.accuracy(Zte, yte), joint=row["acc_star"], within_task=row["acc_dagger"],
                   within_task_fix=de_center(dag, st).accuracy(Zte, yte), slda=row["acc_lda"], ncm=row["acc_ncm"],
                   fecam=FeCAMHead(Ztr, ytr, N).accuracy(Zte, yte), ranpac=rp.accuracy(Zte, yte), eta=eta, tau=tau, K=K_method, schedule=schedule,
                   clip_frac=ours.info.get("clip_frac"), max_grad_norm=ours.info.get("max_grad_norm"), grad_below_half_tau=ours.info.get("grad_below_half_tau"))
    if K_sweep and alpha_row is not None:
        # E_CF(K) = L_test(diagonal head after K clipped steps per task) - L_test(converged diagonal head psi_dagger)
        arms = {"clipped": {}, "unclipped": {}}
        for K in K_sweep:
            eta, tau = _sched(K)
            hc, _ = clipped_sgd_head(Ztr, ytr, toc, tasks, K=K, eta=eta, tau=tau, batch=batch, lam=lam, seed_purpose=f"{tag}:ecf:{K}:clip", device=device)
            hu, _ = clipped_sgd_head(Ztr, ytr, toc, tasks, K=K, eta=eta, tau=np.inf, batch=batch, lam=lam, seed_purpose=f"{tag}:ecf:{K}:unclip", clip=False, device=device)
            arms["clipped"][str(K)] = hc.ce_loss(Zte, yte) - L_dag
            arms.setdefault("clip_frac", {})[str(K)] = hc.info.get("clip_frac"); arms.setdefault("tau", {})[str(K)] = tau; arms.setdefault("eta", {})[str(K)] = eta
            arms["unclipped"][str(K)] = hu.ce_loss(Zte, yte) - L_dag
        row["ecf"] = arms
    return row


def alpha_config(backbone: str, dataset: str, root: Path, seeds=R.SEEDS, lr=0.1, batch=128, device=None) -> dict:
    """app:alpha-estimation: K_warm = 1000 vanilla SGD steps on the full N-way CE of the standardized features
    from a random head; Hill tail index of the gradient norms (k = 50) with a 1000-resample bootstrap CI; the
    curvature probe mu_hat and the scale sigma_hat that Prop 5.1 needs."""
    (Xtr, ytr, _), _, meta = load_pair(backbone, dataset, root)
    N = R.DATASETS[dataset].n_classes
    rows = []
    for s in seeds:
        toc = R.task_of_class(dataset, R.class_ordering(dataset, s))
        st = ClassStats.from_features(Xtr, ytr, toc, protocol="A"); Z = st.transform(Xtr)
        norms, th = warmup_gradient_norms(Z, ytr, N, k_warm=R.K_WARM, batch=batch, lr=lr, seed_purpose=f"warm:{backbone}:{dataset}:{s}", device=device)
        h = hill_estimate(norms, k=R.HILL_K, n_bootstrap=R.N_BOOTSTRAP, seed_purpose=f"hill:{backbone}:{dataset}:{s}")
        mu_hat, _ = curvature_probe(Z, ytr, N, th, 1e-4, seed_purpose=f"probe:{backbone}:{dataset}:{s}")
        a_eff = float(min(h["alpha"], 100.0))     # the moment of order alpha; capped so light-tailed streams cannot overflow
        sigma_hat = float(np.exp((special.logsumexp(a_eff * np.log(np.maximum(norms, 1e-300))) - np.log(len(norms))) / a_eff))
        rows.append(dict(seed=s, alpha=h["alpha"], ci=list(h["ci"]), beta_hat=float(norms.max()), beta_bound=smoothness_bound(Z, 1e-4), mu_hat=mu_hat,
                         sigma_hat=sigma_hat, alpha_used_for_moments=a_eff, warm_lr=lr, k=R.HILL_K))
    return dict(backbone=backbone, dataset=dataset, provenance=meta.get("provenance", "real"),
                alpha=_ms([r["alpha"] for r in rows]), ci_lo=_ms([r["ci"][0] for r in rows]), ci_hi=_ms([r["ci"][1] for r in rows]),
                beta_hat=_ms([r["beta_hat"] for r in rows]), beta_bound=_ms([r["beta_bound"] for r in rows]), mu_hat=_ms([r["mu_hat"] for r in rows]),
                sigma_hat=_ms([r["sigma_hat"] for r in rows]), per_seed=rows)


# ---------------------------------------------------------------------------
# Slope fits (the Theorem 3.5 test on CE heads)
# ---------------------------------------------------------------------------

def _ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(intercept=float(beta[0]), slope=float(beta[1]), r2=1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), ss_res=ss_res, n=int(len(x)))


def _pooled_common_slope(x, y, groups):
    """y = a_g + m x: one slope, one intercept per group."""
    G = np.unique(groups)
    X = np.column_stack([x] + [(groups == g).astype(float) for g in G])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(slope=float(beta[0]), r2=1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), ss_res=ss_res, df=int(len(y) - X.shape[1]), n=int(len(y)),
                intercepts={str(g): float(b) for g, b in zip(G, beta[1:])})


def _separate_slopes(x, y, groups):
    G = np.unique(groups)
    X = np.column_stack([(groups == g).astype(float) for g in G] + [x * (groups == g) for g in G])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return dict(ss_res=float(((y - X @ beta) ** 2).sum()), df=int(len(y) - X.shape[1]))


def slope_fits(tc_results: Dict[Tuple[str, str], dict], anchors: Optional[Dict[str, dict]] = None, n_bootstrap=1000) -> dict:
    """Per-backbone, pooled-12 and pooled-16 slopes of log TC gap on gamma_min, bootstrap CIs over orderings
    (one ordering resampled per configuration), and the backbone-slope interaction F test."""
    cfgs = [(b, d) for b in R.PAPER_A_BACKBONES for d in R.PAPER_A_DATASETS if (b, d) in tc_results]

    def point_set(seed_pick=None):
        xs, ys, gs = [], [], []
        for (b, d) in cfgs:
            per = tc_results[(b, d)]["per_seed"]
            if seed_pick is None:
                g = np.mean([p["gamma_min"] for p in per]); t = np.mean([p["tc_gap"] for p in per])
            else:
                p = per[seed_pick[(b, d)] % len(per)]; g, t = p["gamma_min"], p["tc_gap"]
            if t > 0:
                xs.append(g); ys.append(np.log(t)); gs.append(b)
        return np.asarray(xs, float), np.asarray(ys, float), np.asarray(gs)

    x, y, g = point_set()
    out = dict(per_backbone={}, n_configs=int(len(x)), configs=[f"{b}|{d}" for b, d in cfgs])
    if len(x) < 3:
        out["note"] = "fewer than three configurations with a positive TC gap; no slope fit"
        return out
    for b in R.PAPER_A_BACKBONES:
        m = g == b
        if m.sum() >= 2:
            out["per_backbone"][b] = _ols(x[m], y[m])
    pooled = _pooled_common_slope(x, y, g); out["pooled12"] = pooled
    sep = _separate_slopes(x, y, g)
    df_num = pooled["df"] - sep["df"]
    if sep["df"] > 0 and sep["ss_res"] > 0 and df_num > 0:
        F = ((pooled["ss_res"] - sep["ss_res"]) / df_num) / (sep["ss_res"] / sep["df"])
        out["interaction"] = dict(F=float(F), df=[int(df_num), int(sep["df"])], p=float(sps.f.sf(F, df_num, sep["df"])))
    else:
        out["interaction"] = dict(F=None, df=[int(df_num), int(sep["df"])], p=None)
    rg = R.rng("slope-bootstrap"); boots = {b: [] for b in R.PAPER_A_BACKBONES}; bints = {b: [] for b in R.PAPER_A_BACKBONES}; bp = []
    nseeds = {k: len(v["per_seed"]) for k, v in tc_results.items()}
    for _ in range(n_bootstrap):
        pick = {k: int(rg.integers(nseeds[k])) for k in cfgs}
        xb, yb, gb = point_set(pick)
        for b in R.PAPER_A_BACKBONES:
            m = gb == b
            if m.sum() >= 2:
                f_ = _ols(xb[m], yb[m]); boots[b].append(f_["slope"]); bints[b].append(f_["intercept"])
        if len(xb) >= 3:
            bp.append(_pooled_common_slope(xb, yb, gb)["slope"])
    for b in out["per_backbone"]:
        out["per_backbone"][b]["ci95"] = [float(v) for v in np.percentile(boots[b], [2.5, 97.5])] if boots[b] else None
        # bootstrap lines for the figure's shaded bands (fig:gamma-tc): slope and intercept per resample
        out["per_backbone"][b]["boot_slopes"] = [float(v) for v in boots[b]]; out["per_backbone"][b]["boot_intercepts"] = [float(v) for v in bints[b]]
    out["pooled12"]["ci95"] = [float(v) for v in np.percentile(bp, [2.5, 97.5])] if bp else None
    out["pooled12"]["boot_slopes"] = [float(v) for v in bp]          # the inset of fig:gamma-tc
    if anchors:
        ds = [d for d in anchors if anchors[d]["tc_gap"]["mean"] and anchors[d]["tc_gap"]["mean"] > 0]
        xa = np.array([anchors[d]["gamma_min"]["mean"] for d in ds]); ya = np.log([anchors[d]["tc_gap"]["mean"] for d in ds])
        ga = np.array(["random_vit"] * len(ds))
        out["pooled16"] = _pooled_common_slope(np.concatenate([x, xa]), np.concatenate([y, ya]), np.concatenate([g, ga]))
        bp16 = []
        for _ in range(n_bootstrap):
            pick = {k: int(rg.integers(nseeds[k])) for k in cfgs}
            xb, yb, gb = point_set(pick)
            if len(xb) >= 3:
                bp16.append(_pooled_common_slope(np.concatenate([xb, xa]), np.concatenate([yb, ya]), np.concatenate([gb, ga]))["slope"])
        out["pooled16"]["ci95"] = [float(v) for v in np.percentile(bp16, [2.5, 97.5])] if bp16 else None
        out["pooled16"]["anchors"] = ds
    out["reference"] = dict(operational=-1 / 8, conservative=-1 / 16)
    return out


# ---------------------------------------------------------------------------
# Predictor regression (Section 8 of the paper)
# ---------------------------------------------------------------------------

def predictor_regression(rows: List[dict]) -> dict:
    """rows: dicts with final_acc, gamma_min, probe_acc, ncm_acc, dataset (one per configuration).
    Single-predictor R^2, nested Delta R^2 with F(1, n-4), the reduced model's own F, and leave-one-dataset-out Delta R^2."""
    y = np.array([r["final_acc"] for r in rows], float); n = len(y)

    def r2(cols, yy=y):
        nn = len(yy); X = np.column_stack([np.ones(nn)] + cols); beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        res = yy - X @ beta; sst = float(((yy - yy.mean()) ** 2).sum())
        return (1 - float((res ** 2).sum()) / sst) if sst > 0 else float("nan"), X.shape[1]

    g = np.array([r["gamma_min"] for r in rows], float); p = np.array([r["probe_acc"] for r in rows], float); c = np.array([r["ncm_acc"] for r in rows], float)
    out = dict(n=n, r2_single=dict(gamma=r2([g])[0], probe=r2([p])[0], ncm=r2([c])[0]))
    r2_red, k_red = r2([p, c]); r2_full, k_full = r2([p, c, g]); df2 = n - k_full
    if df2 > 0 and r2_full < 1:
        F = ((r2_full - r2_red) / (k_full - k_red)) / ((1 - r2_full) / df2)
        out.update(F=float(F), p=float(sps.f.sf(F, k_full - k_red, df2)))
    else:
        out.update(F=None, p=None)
    out.update(r2_reduced=r2_red, r2_full=r2_full, delta_r2=r2_full - r2_red, df=[k_full - k_red, df2])
    if n - k_red > 0 and r2_red < 1:
        out["F_reduced"] = float((r2_red / (k_red - 1)) / ((1 - r2_red) / (n - k_red))); out["F_reduced_df"] = [k_red - 1, n - k_red]
    loo = {}
    for d in sorted({r["dataset"] for r in rows}):
        keep = [i for i, r in enumerate(rows) if r["dataset"] != d]
        if len(keep) >= 5:
            yy = y[keep]
            loo[d] = r2([p[keep], c[keep], g[keep]], yy)[0] - r2([p[keep], c[keep]], yy)[0]
    out["loo_delta_r2"] = loo
    return out


def ecf_fits(per_seed_rows: List[dict], K_sweep) -> dict:
    """Log-log OLS slope of mean E_CF(K) on K per arm with a bootstrap CI over orderings.  Budgets whose mean
    E_CF is not positive (the SGD head beat the reference minimizer on the test loss) are excluded from the log
    fit and listed in ``excluded``; a fit needs at least three positive budgets."""
    out = {}
    rg = R.rng("ecf-boot")
    for arm in ("clipped", "unclipped"):
        vals = {str(K): [r["ecf"][arm][str(K)] for r in per_seed_rows if "ecf" in r] for K in K_sweep}
        means = {K: _ms(vals[K]) for K in vals}
        pos = [K for K in K_sweep if means[str(K)]["mean"] is not None and means[str(K)]["mean"] > 0]
        res = dict(E_CF=means, excluded=[K for K in K_sweep if K not in pos])
        if len(pos) >= 3:
            lx = np.log(np.asarray(pos, float)); ly = np.log([means[str(K)]["mean"] for K in pos])
            fit = _ols(lx, ly); boots = []
            nseed = len(per_seed_rows)
            for _ in range(1000):
                pick = rg.integers(0, nseed, nseed)
                mb = [np.mean([vals[str(K)][i] for i in pick]) for K in pos]
                if all(m > 0 for m in mb):
                    boots.append(_ols(lx, np.log(mb))["slope"])
            fit["ci95"] = [float(v) for v in np.percentile(boots, [2.5, 97.5])] if len(boots) >= 10 else None
            res["fit"] = fit
        else:
            res["fit"] = dict(slope=None, note="fewer than three budgets with positive mean E_CF")
        out[arm] = res
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def condition_S_survey(root: Path, out_path: Path, seeds=R.SEEDS, backbones=R.PAPER_A_BACKBONES, datasets=R.PAPER_A_DATASETS, log=print) -> dict:
    """Condition (S) of the camera-ready on every cached configuration, from the feature caches alone (no
    head training; minutes on CPU): gamma_min and its pair, nu_ij in the paper's isotropic ||.||_* metric at
    the task-level and class-level lambda_bar, the metric mismatch delta_ij = gamma_ij - ||mu^(i) - mu^(j)||^2 /
    lambda_bar, the exponent fraction a_ij = (delta_ij + gamma_ij nu_ij)/gamma_ij, whether both clauses hold on
    the worst pair, and the fraction of ordered task pairs on which they hold.  This is the non-vacuity test of
    fix A-01: the 1/16 guarantee of Theorem 3.6 applies to a configuration only where clause_holds is true."""
    out = {}
    for b in backbones:
        for d in datasets:
            if not cache_exists(b, d, root):
                continue
            (Xtr, ytr, _), _, meta = load_pair(b, d, root)
            rows = []
            for s in seeds:
                toc = R.task_of_class(d, R.class_ordering(d, s))
                st = ClassStats.from_features(Xtr, ytr, toc, protocol="A")
                cs = st.condition_S(Xtr, ytr); cs["pair"] = [int(cs["pair"][0]), int(cs["pair"][1])]; cs["seed"] = s
                rows.append(cs)
                log(f"[A] condition-S {b}/{d} s{s}: gamma_min {cs['gamma_min']:.3f} nu {cs['nu']:.3f} delta/gamma {cs['delta_over_gamma']:.3f} "
                    f"a_ij {cs['a_ij']:.3f} holds {cs['clause_holds']} frac_pairs {cs['frac_pairs_clause']:.3f} "
                    f"| class level: nu {cs.get('nu_class', float('nan')):.3f} delta/gamma {cs.get('delta_over_gamma_class', float('nan')):.3f} holds {cs.get('clause_holds_class')} "
                    f"| pooled metric: nu {cs['nu_pooled']:.3f} kappa_sigma {cs['kappa_sigma']:.2f} sqrt(kappa) nu {cs['nu_pooled_scaled']:.3f} holds {cs['clause_holds_pooled']} "
                    f"| pair metric: nu {cs['nu_pair']:.3f} holds {cs['clause_pair']} kappa_cls {cs['kappa_cls']} exponent {cs['exponent_pair']}")
            num = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool) and k != "seed"]
            out[f"{b}|{d}"] = dict(backbone=b, dataset=d, provenance=meta.get("provenance", "real"),
                                   **{k: _ms([r[k] for r in rows]) for k in num},
                                   clause_holds=[bool(r["clause_holds"]) for r in rows],
                                   clause_holds_class=[bool(r.get("clause_holds_class", False)) for r in rows],
                                   clause_holds_pooled=[bool(r["clause_holds_pooled"]) for r in rows],
                                   clause_pair=[bool(r["clause_pair"]) for r in rows], per_seed=rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), configs=out), indent=1, default=str), encoding="utf-8")
    log(f"[A] wrote {out_path}")
    return out


def run_all(root: Path, out_dir: Path, seeds=R.SEEDS, optimizer="adamw", K_method=2000, skip_ecf=False,
            backbones=R.PAPER_A_BACKBONES, datasets=R.PAPER_A_DATASETS, K_sweep=R.K_SWEEP, ecf_backbones=("vit_in21k", "dinov2_b14"),
            ranpac_M=10000, device=None, protocol=None, schedule: str = "printed", log=print) -> dict:
    """Drive every Paper A measurement over the available caches.  ``protocol`` overrides the AdamW head protocol
    (R.HEAD_PROTOCOL["A"]); the selftest shortens its epochs, the real run leaves it alone.  ``schedule`` picks the
    Algorithm 1 stage-0 schedule for the clipped-SGD arms: "printed" (the paper's Proposition 5.3 as printed) or "consistent"
    (the units-consistent reading); both are recorded in the rows."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol or R.HEAD_PROTOCOL["A"]
    dev = _backend.describe(device)
    cfgs = [(b, d) for b in backbones for d in datasets if cache_exists(b, d, root)]
    missing = [(b, d) for b in backbones for d in datasets if not cache_exists(b, d, root)]
    t0 = time.time()
    log(f"[A] backend {dev}; {len(cfgs)} configurations, {len(missing)} missing: {missing}")
    # pre-pass: tail index per backbone on CIFAR-100 (Algorithm 1 stage 0)
    alpha = {}
    for b in backbones:
        if cache_exists(b, "cifar100", root):
            alpha[b] = alpha_config(b, "cifar100", root, seeds, device=device)
            log(f"[A] alpha {b}: {alpha[b]['alpha']['mean']:.3f} (CI lo {alpha[b]['ci_lo']['mean']:.2f} hi {alpha[b]['ci_hi']['mean']:.2f}) ({time.time()-t0:.0f}s)")
    gamma, tc, method, ecf = {}, {}, {}, {}
    for (b, d) in cfgs:
        (Xtr, ytr, _), (Xte, yte, _), meta = load_pair(b, d, root)
        prov = meta.get("provenance", "real"); rows = []
        do_ecf = (not skip_ecf) and b in ecf_backbones and d == "cifar100" and b in alpha
        for si, s in enumerate(seeds):
            a_row = alpha[b]["per_seed"][si % len(alpha[b]["per_seed"])] if b in alpha else None
            rows.append(seed_pass(b, d, s, Xtr, ytr, Xte, yte, optimizer=optimizer, protocol=protocol, alpha_row=a_row, K_method=K_method,
                                  K_sweep=(K_sweep if do_ecf else None), ranpac_M=ranpac_M, device=device, schedule=schedule))
            log(f"[A] {b}/{d} s{s}: gamma {rows[-1]['gamma_min']:.3f} (eq1 {rows[-1]['gamma_min_eq1']:.3f}) tc_gap {rows[-1]['tc_gap']:.4f} acc star {rows[-1]['acc_star']:.4f}"
                + (f" ours {rows[-1]['ours']:.4f}" if "ours" in rows[-1] else "") + f" ({time.time()-t0:.0f}s)")
        base = dict(backbone=b, dataset=d, provenance=prov, optimizer=optimizer, backend=dev["backend"])
        cond_keys = [k for k in rows[0]["condition_S"] if isinstance(rows[0]["condition_S"][k], (int, float)) and not isinstance(rows[0]["condition_S"][k], bool)]
        gamma[(b, d)] = dict(base, **_agg(rows, ["gamma_min", "gamma_min_eq1", "nu", "rhs", "lam_lw", "gamma_min_unshrunk"]),
                             condition_S={k: _ms([r["condition_S"][k] for r in rows]) for k in cond_keys},
                             condition_S_clause_holds={k: [bool(r["condition_S"][k]) for r in rows] for k in rows[0]["condition_S"] if k.startswith("clause_holds")},
                             per_seed=[{k: r[k] for k in ("seed", "gamma_min", "gamma_min_eq1", "pair", "nu", "rhs", "lam_lw", "gamma_min_unshrunk", "condition_S")} for r in rows])
        tc[(b, d)] = dict(base, **_agg(rows, ["tc_gap", "L_dagger", "L_star", "acc_dagger", "acc_star", "acc_lda", "acc_ncm", "gamma_min"]),
                          log_tc_gap=_ms([np.log(r["tc_gap"]) if r["tc_gap"] > 0 else np.nan for r in rows]),
                          fit_max_grad_dagger=_ms([max(f["max_grad"] for f in r["fit_dagger"]) for r in rows if r["fit_dagger"]]),
                          fit_max_grad_star=_ms([r["fit_star"]["max_grad"] for r in rows]),
                          per_seed=[{k: r[k] for k in ("seed", "tc_gap", "L_dagger", "L_star", "acc_dagger", "acc_star", "acc_lda", "acc_ncm", "gamma_min", "fit_dagger", "fit_star")} for r in rows])
        if all("ours" in r for r in rows):
            mkeys = ["ours", "ours_classil", "ours_fix", "ours_ce", "sequential_head", "joint", "within_task", "within_task_fix", "slda", "ncm", "fecam", "ranpac", "avg_incremental"]
            method[(b, d)] = dict(base, K=K_method, ranpac_M=ranpac_M, **_agg(rows, mkeys),
                                  per_seed=[{k: r.get(k) for k in ["seed"] + mkeys + ["ours_stages", "eta", "tau", "schedule", "clip_frac", "max_grad_norm", "grad_below_half_tau"]} for r in rows])
        if do_ecf:
            ecf[b] = dict(base, K_sweep=list(K_sweep), arms=ecf_fits(rows, K_sweep),
                          per_seed=[dict(seed=r["seed"], ecf=r["ecf"]) for r in rows])
            log(f"[A] ecf {b}: clipped slope {ecf[b]['arms']['clipped']['fit']['slope']} unclipped {ecf[b]['arms']['unclipped']['fit']['slope']}")
    anchors = {}
    for d in datasets:
        if cache_exists("random_vit", d, root):
            (Xtr, ytr, _), (Xte, yte, _), meta = load_pair("random_vit", d, root)
            rows = [seed_pass("random_vit", d, s, Xtr, ytr, Xte, yte, optimizer=optimizer, protocol=protocol, with_methods=False, device=device) for s in seeds]
            anchors[d] = dict(gamma_min=_ms([r["gamma_min"] for r in rows]), tc_gap=_ms([r["tc_gap"] for r in rows]),
                              log_tc_gap=_ms([np.log(r["tc_gap"]) if r["tc_gap"] > 0 else np.nan for r in rows]),
                              acc_star=_ms([r["acc_star"] for r in rows]), provenance=meta.get("provenance", "real"))
            log(f"[A] anchor random_vit/{d}: gamma {anchors[d]['gamma_min']['mean']:.3f} tc_gap {anchors[d]['tc_gap']['mean']:.4f} ({time.time()-t0:.0f}s)")
    slopes = slope_fits(tc, anchors or None) if tc else None
    prow = [dict(backbone=b, dataset=d, final_acc=method[(b, d)]["ours"]["mean"], gamma_min=gamma[(b, d)]["gamma_min"]["mean"],
                 probe_acc=method[(b, d)]["joint"]["mean"], ncm_acc=method[(b, d)]["ncm"]["mean"]) for (b, d) in cfgs if (b, d) in method]
    predictor = predictor_regression(prow) if len(prow) >= 6 else None

    def keyed(dct): return {f"{k[0]}|{k[1]}": v for k, v in dct.items()}
    payload = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), optimizer=optimizer, protocol=protocol, seeds=list(seeds), backend=dev,
                   schedule=schedule, configs=[f"{b}|{d}" for b, d in cfgs], missing=[f"{b}|{d}" for b, d in missing],
                   gamma=keyed(gamma), tc=keyed(tc), alpha=alpha, method=keyed(method), ecf=ecf, anchors=anchors,
                   slopes=slopes, predictor=predictor, predictor_rows=prow,
                   provenance=sorted({v["provenance"] for v in gamma.values()} | {v["provenance"] for v in anchors.values()}))
    (out_dir / "paperA_results.json").write_text(json.dumps(payload, indent=1, default=float))
    log(f"[A] wrote {out_dir / 'paperA_results.json'} in {time.time()-t0:.0f}s")
    return payload
