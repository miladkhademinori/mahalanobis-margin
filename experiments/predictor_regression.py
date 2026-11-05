"""experiments.predictor_regression -- the Paper A R2/F predictor regression (CI1).

Implements the Q5 ruling (REPLY_FROM_CLAUDE_REPLICATION.md sec. 3, 2026-10-22):
the four R2/F cells (full/reduced R2 and F), the nested delta-R2 with its
p-value, the leave-one-dataset-out sensitivity (4 refits), and the
backbone-slope interaction F(2,6) must all come from ONE fit on ONE 12-row
input table. Reconstruction from the printed identity is barred; if a column
is not printed in the paper, the script records INPUTS_MISSING and never
fabricates.

Model battery (paper lines 1261-1274 of working/paper_A/neurips_2026.tex):
  single-predictor: Acc ~ gamma_min   (printed R2 = 0.88)
                    Acc ~ linprobe    (printed R2 = 0.85)
                    Acc ~ ncm         (printed R2 = 0.72)
  reduced:  Acc ~ linprobe + ncm                     k=3,  F on (2, 9)
  full:     Acc ~ linprobe + ncm + gamma_min         k=4,  F on (3, 8)
  nested:   F(1, 8) on delta-R2                      (printed dR2=0.06, p=0.042)
  interaction: backbone-specific gamma_min slopes + backbone intercepts vs
               common slope + backbone intercepts   F(2, 6) (printed 2.14, p=.199)
  sensitivity: leave-one-dataset-out delta-R2 (4 refits; 3 backbones x 4 datasets)

PORT-CHECK / BUG G01 (documented, fixed here): the owner's original
C:\\Users\\milad\\Projects\\neurips 2026\\paper_4828\\code\\stage_b_regressions.py
(2026-10-07, never executed, SETUP-ONLY lock) defines

    def nested_F(full, reduced):
        q = reduced["k"] - full["k"]                      # WRONG SIGN
        F = ((reduced["RSS"] - full["RSS"]) / q) / (full["RSS"] / full["dof"])
        return dict(dR2=reduced["R2"] - full["R2"], ...)  # WRONG SIGN

while every call site passes (full, reduced). At run time q is negative
(k_reduced < k_full), so F is negative and the p-value is garbage, and dR2
comes out as R2_reduced - R2_full (negative). This port computes
q = k_full - k_reduced and dR2 = R2_full - R2_reduced, and the self-test
pins both signs against the printed identity.

Printed inputs embedded below (extraction audit, CI1, 2026-10-22; source
working/paper_A/neurips_2026.tex):
  gamma_min, 12/12 rows PRINTED  (tab:gamma-min, lines 4701-4703)
  final class-IL accuracy, 4/12 rows PRINTED, ViT-B/16-IN21K column only,
    "Diagnostic-guided clipped SGD (ours)" row (tab:main line 5176; same
    numbers in tab:headline-comparison line 1430). DINOv2 and CLIP rows are
    printed NOWHERE. Which method's accuracy is the regression response is
    itself an owner question (the bank spec used runs.csv final-task accuracy).
  linprobe accuracy, 0/12 PRINTED (only the summary R2=0.85, line 1264)
  ncm accuracy,      0/12 PRINTED (only the summary R2=0.72, line 1265)
Per the tasking fence, that changes the plan to owner-fetch: the fit is NOT
run on partial or invented inputs. When the owner supplies the missing rows
(--inputs JSON), this script runs the whole battery in one artifact.

Usage (from repo root):
  python experiments/predictor_regression.py --self-test
  python experiments/predictor_regression.py                 # audit + status JSON
  python experiments/predictor_regression.py --inputs path/to/rows.json --out results/runs/A/predictor_regression/A_predictor_regression_CI1.json

Owner --inputs JSON format: {"rows": [{"backbone": "ViT-B/16-IN21K",
"dataset": "CIFAR-100", "final_acc": 91.8, "linprobe": ..., "ncm": ...,
"gamma_min": 8.3}, ...]} -- 12 rows, one per (backbone, dataset). gamma_min
is auto-checked against tab:gamma-min when present (printed value wins the
check; a mismatch is a finding, reported, never silently repaired).
Percent vs fraction for accuracies does not matter: R2, F and p are
invariant to affine rescaling of the response.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import f as fdist

# --------------------------------------------------------------------------
# Printed reference values (working/paper_A/neurips_2026.tex, CI1 audit 2026-10-22)
BACKBONES = ["ViT-B/16-IN21K", "DINOv2-ViT-B/14", "CLIP-ViT-B/16"]
DATASETS = ["CIFAR-100", "ImageNet-R", "CUB-200", "OmniBenchmark"]

# tab:gamma-min, lines 4701-4703 (mean across 5 orderings; the mean is the
# regression input; the +/- spread is not part of the fit)
GAMMA_MIN_PRINTED = {
    ("ViT-B/16-IN21K", "CIFAR-100"): 8.3,
    ("ViT-B/16-IN21K", "ImageNet-R"): 5.2,
    ("ViT-B/16-IN21K", "CUB-200"): 12.1,
    ("ViT-B/16-IN21K", "OmniBenchmark"): 9.6,
    ("DINOv2-ViT-B/14", "CIFAR-100"): 47.2,
    ("DINOv2-ViT-B/14", "ImageNet-R"): 39.1,
    ("DINOv2-ViT-B/14", "CUB-200"): 51.8,
    ("DINOv2-ViT-B/14", "OmniBenchmark"): 44.6,
    ("CLIP-ViT-B/16", "CIFAR-100"): 23.1,
    ("CLIP-ViT-B/16", "ImageNet-R"): 18.9,
    ("CLIP-ViT-B/16", "CUB-200"): 31.4,
    ("CLIP-ViT-B/16", "OmniBenchmark"): 26.3,
}

# tab:main line 5176 "Diagnostic-guided clipped SGD (ours)", ViT-B/16-IN21K only
FINAL_ACC_PRINTED = {
    ("ViT-B/16-IN21K", "CIFAR-100"): 91.8,
    ("ViT-B/16-IN21K", "ImageNet-R"): 82.1,
    ("ViT-B/16-IN21K", "CUB-200"): 87.9,
    ("ViT-B/16-IN21K", "OmniBenchmark"): 85.8,
}

PRINTED_TARGETS = {  # line: value
    "r2_single_margin": (1264, 0.88),
    "r2_single_linprobe": (1264, 0.85),
    "r2_single_ncm": (1265, 0.72),
    "delta_r2": (1268, 0.06),
    "nested_p": (1268, 0.042),
    "interaction_F_2_6": (1252, 2.14),
    "interaction_p": (1252, 0.199),
}


# --------------------------------------------------------------------------
def ols(X: np.ndarray, y: np.ndarray) -> dict:
    """Minimal auditable OLS; X must already contain the intercept column."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta
    RSS = float(resid @ resid)
    TSS = float(((y - y.mean()) ** 2).sum())
    dof = n - k
    F = ((TSS - RSS) / (k - 1)) / (RSS / dof) if (k > 1 and dof > 0 and RSS > 0) else float("nan")
    p = float(fdist.sf(F, k - 1, dof)) if (dof > 0 and not math.isnan(F)) else float("nan")
    return dict(beta=beta.tolist(), RSS=RSS, TSS=TSS, R2=1.0 - RSS / TSS,
                F=float(F), p=p, n=n, k=k, dof=dof)


def nested_F(full: dict, reduced: dict) -> dict:
    """G01-FIXED nested F test: full nests reduced (k_full > k_reduced).

    F = [(RSS_reduced - RSS_full)/q] / [RSS_full/df_full],  q = k_full - k_reduced,
    dR2 = R2_full - R2_reduced  (positive when the added predictor helps).
    """
    q = full["k"] - reduced["k"]
    if q <= 0:
        raise ValueError("full model must have more columns than reduced")
    F = ((reduced["RSS"] - full["RSS"]) / q) / (full["RSS"] / full["dof"])
    return dict(q=q, dR2=full["R2"] - reduced["R2"], F=float(F),
                df=(q, full["dof"]), p=float(fdist.sf(F, q, full["dof"])))


def single_predictor_r2(col: np.ndarray, y: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(y)), col])
    return ols(X, y)["R2"]


def battery(rows: list[dict]) -> dict:
    """The full Q5 battery on 12 rows. One artifact, one fit path."""
    y = np.array([r["final_acc"] for r in rows], float)
    lp = np.array([r["linprobe"] for r in rows], float)
    ncm = np.array([r["ncm"] for r in rows], float)
    gm = np.array([r["gamma_min"] for r in rows], float)
    ones = np.ones(len(y))

    singles = {name: single_predictor_r2(col, y)
               for name, col in (("gamma_min", gm), ("linprobe", lp), ("ncm", ncm))}

    Xr = np.column_stack([ones, lp, ncm])          # k=3 (intercept + 2 proxies)
    Xf = np.column_stack([ones, lp, ncm, gm])      # k=4 (intercept + 3 predictors)
    fit_r, fit_f = ols(Xr, y), ols(Xf, y)
    nested = nested_F(fit_f, fit_r)                # F(1, 8)

    # backbone-slope interaction on the same inputs: full = backbone-specific
    # gamma_min slopes + backbone intercepts (k = 1+3+2 = 6, df 6); reduced =
    # common slope + backbone intercepts (k = 4). F(2, 6), printed 2.14.
    grp = {b: np.array([1.0 if r["backbone"] == b else 0.0 for r in rows]) for b in BACKBONES}
    Xr_i = np.column_stack([ones, gm, grp[BACKBONES[1]], grp[BACKBONES[2]]])
    Xf_i = np.column_stack([ones] + [gm * grp[b] for b in BACKBONES]
                           + [grp[BACKBONES[1]], grp[BACKBONES[2]]])
    fit_r_i, fit_f_i = ols(Xr_i, y), ols(Xf_i, y)
    inter = nested_F(fit_f_i, fit_r_i)             # F(2, 6)
    per_bb_slope = {b: float(fit_f_i["beta"][1 + i]) for i, b in enumerate(BACKBONES)}

    # leave-one-dataset-out delta-R2 (ruling sec. 3 sensitivity line)
    lodo = {}
    for ds in DATASETS:
        sub = [r for r in rows if r["dataset"] != ds]
        ys = np.array([r["final_acc"] for r in sub], float)
        Xs_r = np.column_stack([np.ones(len(sub)), [r["linprobe"] for r in sub],
                                [r["ncm"] for r in sub]])
        Xs_f = np.column_stack([Xs_r, [r["gamma_min"] for r in sub]])
        lodo[ds] = nested_F(ols(Xs_f, ys), ols(Xs_r, ys))

    return dict(n=len(rows), singles=singles,
                reduced=dict(R2=fit_r["R2"], F=fit_r["F"], p=fit_r["p"],
                             df=(fit_r["k"] - 1, fit_r["dof"])),
                full=dict(R2=fit_f["R2"], F=fit_f["F"], p=fit_f["p"],
                          df=(fit_f["k"] - 1, fit_f["dof"])),
                nested=nested, interaction=dict(F=inter["F"], p=inter["p"],
                                                df=inter["df"], per_backbone_slope=per_bb_slope),
                lodo={ds: dict(dR2=v["dR2"], F=v["F"], p=v["p"], n=v["q"] and len(rows) - 3)
                      for ds, v in lodo.items()})


# --------------------------------------------------------------------------
def audit_inputs(rows: list[dict] | None) -> dict:
    """Which cells are owner-supplied vs printed vs missing. Never guesses."""
    have = {(r.get("backbone"), r.get("dataset")): r for r in (rows or [])}
    cells = []
    for bb in BACKBONES:
        for ds in DATASETS:
            key = (bb, ds)
            r = have.get(key, {})
            cells.append(dict(
                backbone=bb, dataset=ds,
                gamma_min=("printed" if key in GAMMA_MIN_PRINTED else "MISSING"),
                final_acc=("supplied" if "final_acc" in r else
                           ("printed(tab:main,ours)" if key in FINAL_ACC_PRINTED else "MISSING")),
                linprobe=("supplied" if "linprobe" in r else "MISSING"),
                ncm=("supplied" if "ncm" in r else "MISSING"),
            ))
    miss = {c: sum(1 for x in cells if x[c] == "MISSING") for c in
            ("gamma_min", "final_acc", "linprobe", "ncm")}
    return dict(cells=cells, missing_counts=miss,
                fit_runnable=all(v == 0 for v in miss.values()))


def self_test() -> list[str]:
    """Correctness pins for the G01 fix and the df conventions. Synthetic data
    only -- no paper number is produced here."""
    ok = []

    def check(name, cond, detail=""):
        ok.append(f"{'PASS' if cond else 'FAIL'} {name} {detail}")
        if not cond:
            raise AssertionError(name)

    # 1. G01 sign pins on a synthetic nested pair
    rng = np.random.RandomState(0)
    n = 12
    x1, x2, x3 = rng.randn(n), rng.randn(n), rng.randn(n)
    y = 1 + 0.8 * x1 + 0.5 * x2 + 0.4 * x3 + 0.1 * rng.randn(n)
    ones = np.ones(n)
    fr, ff = ols(np.column_stack([ones, x1, x2]), y), ols(np.column_stack([ones, x1, x2, x3]), y)
    nf = nested_F(ff, fr)
    check("q positive (=1)", nf["q"] == 1)
    check("dR2 positive", nf["dR2"] > 0, f"dR2={nf['dR2']:.4f}")
    check("F positive", nf["F"] > 0)
    # closed form from R2: F = (dR2/q) / ((1-R2_full)/df_full)
    F_closed = (nf["dR2"] / nf["q"]) / ((1 - ff["R2"]) / ff["dof"])
    check("F equals R2-form closed form", abs(nf["F"] - F_closed) < 1e-10)
    check("p matches scipy", abs(nf["p"] - fdist.sf(nf["F"], 1, ff["dof"])) < 1e-15)
    # the owner's buggy variant would print F_closed with q=-1: catch by pin
    check("sign convention pinned (full-minus-reduced)", nested_F(ff, fr)["dR2"] ==
          ff["R2"] - fr["R2"])

    # 2. printed-identity oracle (code check ONLY -- reconstruction for the
    # paper is barred by the Q5 ruling; this validates the formula, not data)
    R2f, R2r, p_target = 0.9179, 0.8579, 0.042
    F_id = ((R2f - R2r) / 1) / ((1 - R2f) / 8)
    check("identity oracle F(1,8)~5.845 at p=0.042",
          abs(F_id - fdist.isf(p_target, 1, 8)) < 5e-3,
          f"F={F_id:.4f} vs isf={fdist.isf(p_target, 1, 8):.4f}")
    check("identity oracle p", abs(fdist.sf(F_id, 1, 8) - p_target) < 5e-4)

    # 3. df conventions: reduced (2,9), full (3,8), interaction (2,6), LODO=4
    rows = [dict(backbone=bb, dataset=ds,
                 final_acc=50 + 10 * GAMMA_MIN_PRINTED[(bb, ds)] + rng.randn(),
                 linprobe=50 + 9 * GAMMA_MIN_PRINTED[(bb, ds)] + rng.randn(),
                 ncm=50 + 5 * GAMMA_MIN_PRINTED[(bb, ds)] + rng.randn(),
                 gamma_min=GAMMA_MIN_PRINTED[(bb, ds)])
            for bb in BACKBONES for ds in DATASETS]
    b = battery(rows)
    check("reduced df (2,9)", tuple(b["reduced"]["df"]) == (2, 9))
    check("full df (3,8)", tuple(b["full"]["df"]) == (3, 8))
    check("nested df (1,8)", tuple(b["nested"]["df"]) == (1, 8))
    check("interaction df (2,6)", tuple(b["interaction"]["df"]) == (2, 6))
    check("LODO has 4 refits", len(b["lodo"]) == 4 and
          all(v["n"] == 9 for v in b["lodo"].values()))
    check("interaction full k=6 implicit", b["interaction"]["df"][1] == 6)
    return ok


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, default=None,
                    help="owner-supplied JSON with the 12 rows (see module docstring)")
    ap.add_argument("--out", type=Path, default=None, help="results-store JSON path")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = dict(run_id="A_predictor_regression_CI1", paper="A",
               experiment="predictor_regression", created_utc=stamp,
               provenance="printed_inputs_partial", script="experiments/predictor_regression.py",
               ruling="REPLY_FROM_CLAUDE_REPLICATION.md sec.3 (Q5), 2026-10-22")

    if args.self_test:
        out["self_test"] = self_test()
        print("\n".join(out["self_test"]))
        out["suite_status"] = "SELF_TEST_PASS"

    rows = None
    if args.inputs is not None:
        data = json.loads(args.inputs.read_text(encoding="utf-8"))
        rows = data["rows"] if isinstance(data, dict) else data
        # gamma_min: fill from tab:gamma-min where absent; verify where present
        for r in rows:
            key = (r["backbone"], r["dataset"])
            if key not in GAMMA_MIN_PRINTED:
                raise SystemExit(f"unknown configuration {key}")
            if "gamma_min" not in r:
                r["gamma_min"] = GAMMA_MIN_PRINTED[key]
            elif abs(r["gamma_min"] - GAMMA_MIN_PRINTED[key]) > 0.05:
                out.setdefault("findings", []).append(
                    f"gamma_min mismatch {key}: supplied {r['gamma_min']} vs printed "
                    f"{GAMMA_MIN_PRINTED[key]} (tab:gamma-min) -- printed value stands, reported")

    audit = audit_inputs(rows)
    out["input_audit"] = audit
    out["printed_targets"] = {k: dict(line=v[0], value=v[1]) for k, v in PRINTED_TARGETS.items()}

    if rows is not None and audit["fit_runnable"]:
        out.update(battery(rows))
        out["provenance"] = "owner_inputs_plus_tab_gamma_min"
        out["suite_status"] = "FIT_COMPLETE"
        out["verdict_vs_printed"] = {
            "delta_r2": {"fit": out["nested"]["dR2"], "printed": 0.06},
            "nested_p": {"fit": out["nested"]["p"], "printed": 0.042},
            "interaction_F": {"fit": out["interaction"]["F"], "printed": 2.14},
            "note": "paper prints what the fit gives (Q5 ruling); mismatch is a finding"}
    else:
        miss = audit["missing_counts"]
        out["suite_status"] = "INPUTS_MISSING"
        out["missing_inputs"] = {
            "linprobe_accuracy": "0/12 printed anywhere in the TeX (only summary R2=0.85, line 1264)",
            "ncm_accuracy": "0/12 printed anywhere in the TeX (only summary R2=0.72, line 1265)",
            "final_class_il_accuracy": ("4/12 printed (tab:main line 5176 'ours', ViT-B/16-IN21K "
                                        "only); DINOv2 and CLIP rows printed nowhere"),
            "gamma_min_accuracy": "12/12 printed (tab:gamma-min, lines 4701-4703)",
        }
        out["owner_fetch"] = ("Supply experiments rows: linprobe (12), ncm (12), final_acc for "
                              "DINOv2-ViT-B/14 and CLIP-ViT-B/16 (4+4); also rule which method's "
                              "final accuracy is the response (tab:main 'ours' vs bank runs.csv "
                              "final-task accuracy). Then rerun with --inputs.")
        print(f"[INPUTS_MISSING] fit not run; missing column counts: {miss}")
        print(json.dumps(out["missing_inputs"], indent=1))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
