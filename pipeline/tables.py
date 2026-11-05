"""pipeline.tables -- LaTeX table bodies for the paper and the paper-vs-reproduced diff.

Every emitter reads only the JSON written by ``paper_a.run_all``
and writes ``tables/out/<label>.tex`` (the tabular rows, ready to paste between the
paper's \\toprule and \\bottomrule) plus ``tables/out/DIFF.md``, which lists every
printed number next to the reproduced one.  Synthetic-provenance results are
rendered with a loud SYNTHETIC watermark and never without it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pipeline import registry as R


def _pm(ms: dict, scale=1.0, nd=1) -> str:
    if not ms or ms.get("mean") is None:
        return "--"
    return f"${ms['mean'] * scale:.{nd}f} \\pm {ms['sd'] * scale:.{nd}f}$"


def _mark(payload: dict) -> str:
    prov = payload.get("provenance", [])
    return "" if prov == ["real"] else f"% PROVENANCE {prov}: NOT A PAPER NUMBER\n"


# ---------------------------------------------------------------------------
# Paper A
# ---------------------------------------------------------------------------

def tab_gamma_min(A: dict) -> str:
    lines = [_mark(A), "% gamma_min by the app:gamma-computation procedure (one pooled shrunk covariance); the eq:gamma variant with"
                       " per-task covariances follows each row as a comment"]
    for b in R.PAPER_A_BACKBONES:
        cells = [_pm(A["gamma"].get(f"{b}|{d}", {}).get("gamma_min")) for d in R.PAPER_A_DATASETS]
        eq1 = [_pm(A["gamma"].get(f"{b}|{d}", {}).get("gamma_min_eq1")) for d in R.PAPER_A_DATASETS]
        lines.append(f"{R.BACKBONES[b].label} & " + " & ".join(cells) + " \\\\ % eq:gamma per-task covariances: " + " | ".join(eq1))
    return "\n".join(lines)


def tab_random_anchor(A: dict) -> str:
    lines = [_mark(A)]
    for d in R.PAPER_A_DATASETS:
        a = A["anchors"].get(d)
        if not a: lines.append(f"{R.DATASETS[d].label} & -- & -- & -- \\\\"); continue
        lines.append(f"{R.DATASETS[d].label} & {_pm(a['gamma_min'], nd=2)} & {_pm(a['tc_gap'], nd=3)} & {_pm(a['log_tc_gap'], nd=2)} \\\\")
    return "\n".join(lines)


def tab_residual_check(A: dict) -> str:
    lines = [_mark(A)]
    for b in R.PAPER_A_BACKBONES:
        for d in R.PAPER_A_DATASETS:
            g = A["gamma"].get(f"{b}|{d}")
            if not g: continue
            nu = g["nu"]["mean"]; flag = "^\\dagger" if nu > 0.5 else ""
            lines.append(f"{R.BACKBONES[b].label} & {R.DATASETS[d].label} & ${g['gamma_min']['mean']:.1f}$ & ${g['rhs']['mean']:.1f}$ & ${nu:.2f}{flag}$ \\\\")
            cs = g.get("condition_S") or {}
            if "delta_over_gamma" in cs:
                holds = g.get("condition_S_clause_holds", {}).get("clause_holds", [])
                lines.append(f"%   Condition (S) clause for the pair: delta/gamma = {cs['delta_over_gamma']['mean']:.3f}, a_ij = {cs['a_ij']['mean']:.3f}"
                             f" (needs nu <= 1/2 and a_ij <= 1/2); holds on {sum(holds)}/{len(holds)} seeds; task-level lambda_bar = {cs['lambda_bar']['mean']:.3g};"
                             f" class-level nu = {cs.get('nu_class', {}).get('mean', float('nan')):.2f}, delta/gamma = {cs.get('delta_over_gamma_class', {}).get('mean', float('nan')):.3f};"
                             f" fraction of all pairs satisfying the clause = {cs['frac_pairs_clause']['mean']:.3f}")
    return "\n".join(lines)


def tab_slope_fits(A: dict) -> str:
    s = A.get("slopes") or {}
    lines = [_mark(A)]
    for b, f in s.get("per_backbone", {}).items():
        ci = f.get("ci95") or [np.nan, np.nan]
        lines.append(f"{R.BACKBONES[b].label} & ${f['slope']:.3f}$ & $[{ci[0]:.3f}, {ci[1]:.3f}]$ & ${f['r2']:.2f}$ \\\\")
    for key, label in (("pooled12", "Pooled (12 pretrained)"), ("pooled16", "Pooled (12 + 4 random anchors)")):
        if key in s:
            f = s[key]; ci = f.get("ci95") or [np.nan, np.nan]
            lines.append(f"{label} & ${f['slope']:.3f}$ & $[{ci[0]:.3f}, {ci[1]:.3f}]$ & ${f['r2']:.2f}$ \\\\")
    i = s.get("interaction")
    if i and i.get("F") is not None:
        lines.append(f"% backbone-slope interaction F({i['df'][0]},{i['df'][1]}) = {i['F']:.2f}, p = {i['p']:.3f}")
    return "\n".join(lines)


def tab_ecf_rates(A: dict) -> str:
    lines = [_mark(A)]
    for arm, title in (("clipped", "Clipped SGD"), ("unclipped", "Unclipped SGD")):
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for b in ("vit_in21k", "dinov2_b14"):
            e = A["ecf"].get(b)
            if not e: continue
            arm_d = e["arms"][arm]; cells = [f"${arm_d['E_CF'][str(K)]['mean']:.3f}$" for K in e["K_sweep"]]
            f = arm_d["fit"]
            if f.get("slope") is None:
                slope = "--"
            elif f.get("ci95"):
                slope = f"${f['slope']:.3f}\\;[{f['ci95'][0]:.3f},\\,{f['ci95'][1]:.3f}]$"
            else:
                slope = f"${f['slope']:.3f}$"
            lines.append(f"{R.BACKBONES[b].label} & " + " & ".join(cells) + f" & {slope} \\\\")
    return "\n".join(lines)


def tab_alpha_fits(A: dict) -> str:
    lines = [_mark(A)]
    for b in R.PAPER_A_BACKBONES:
        a = A["alpha"].get(b)
        if not a: continue
        lines.append(f"{R.BACKBONES[b].label} & ${a['alpha']['mean']:.2f}$ & $[{a['ci_lo']['mean']:.2f},\\,{a['ci_hi']['mean']:.2f}]$ \\\\")
    return "\n".join(lines)


def tab_method_comparison(A: dict) -> str:
    """Frozen-feature rows of tab:main that this code reproduces; the published-method rows are not run here."""
    lines = [_mark(A)]
    rows = [("sequential_head", "Sequential head (lower, frozen features)"), ("joint", "Joint training (upper)"),
            ("slda", "SLDA"), ("ncm", "SimpleCIL / NCM"), ("fecam", "FeCAM-style"), ("ranpac", "RanPAC-style"),
            ("within_task", "Within-task softmax (AdamW)"), ("within_task_fix", "Within-task + de-centering (post-hoc fix)"),
            ("ours", "Diagnostic-guided clipped SGD (ours; diagonal head, Alg. 1)"), ("ours_fix", "ours + de-centering"),
            ("ours_classil", "ours, class-IL reading (CE over seen classes)")]
    for key, label in rows:
        cells = []
        for d in R.PAPER_A_DATASETS:
            m = A["method"].get(f"vit_in21k|{d}")
            cells.append(_pm(m[key], 100, 1) if m else "--")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def predictor_text(A: dict) -> str:
    p = A.get("predictor")
    if not p: return "% predictor regression: not enough configurations"
    def f3(v, nd=3): return "--" if v is None else f"{v:.{nd}f}"
    return (f"% single-predictor R2: gamma {f3(p['r2_single']['gamma'], 2)}, probe {f3(p['r2_single']['probe'], 2)}, ncm {f3(p['r2_single']['ncm'], 2)}\n"
            f"% nested: R2_full {f3(p['r2_full'], 4)} (F={f3(p.get('F'))} on {p['df']}), R2_reduced {f3(p['r2_reduced'], 4)} "
            f"(F={f3(p.get('F_reduced'), 2)} on {p.get('F_reduced_df')}), Delta R2 {f3(p['delta_r2'], 4)}, p = {f3(p.get('p'), 4)}; "
            "leave-one-dataset-out Delta R2: " + ", ".join(f"{d}: {v:.3f}" for d, v in p["loo_delta_r2"].items()))


# ---------------------------------------------------------------------------
# Diff against the printed values
# ---------------------------------------------------------------------------

def diff_report(A: Optional[dict]) -> str:
    rows = ["# Printed vs reproduced", "", "| table | cell | printed | reproduced | note |", "|---|---|---|---|---|"]
    if A:
        for (b, d), (m, s) in R.PRINTED["A.tab:gamma-min"].items():
            g = A["gamma"].get(f"{b}|{d}")
            rows.append(f"| tab:gamma-min | {b}/{d} | {m} ± {s} | {_pm(g['gamma_min']) if g else '--'} | eq:gamma per-task-cov variant {_pm(g.get('gamma_min_eq1')) if g else '--'} |")
        for d, (g0, t0) in R.PRINTED["A.tab:random-anchor"].items():
            a = A["anchors"].get(d)
            rows.append(f"| tab:random-anchor | {d} | γ {g0}, TC {t0} | {('γ ' + _pm(a['gamma_min'], nd=2) + ', TC ' + _pm(a['tc_gap'], nd=3)) if a else '--'} | |")
        s = A.get("slopes") or {}
        for b, v in R.PRINTED["A.slopes"]["per_backbone"].items():
            f = s.get("per_backbone", {}).get(b)
            rows.append(f"| slopes | {b} | {v} | {f['slope']:.3f} | R2 {f['r2']:.2f} |" if f else f"| slopes | {b} | {v} | -- | |")
        if "pooled12" in s: rows.append(f"| slopes | pooled 12 | {R.PRINTED['A.slopes']['pooled12']} | {s['pooled12']['slope']:.3f} {s['pooled12'].get('ci95')} | |")
        if "pooled16" in s: rows.append(f"| slopes | pooled 16 | {R.PRINTED['A.slopes']['pooled16']} | {s['pooled16']['slope']:.3f} {s['pooled16'].get('ci95')} | |")
        if s.get("interaction", {}).get("F") is not None: rows.append(f"| slopes | interaction F | {R.PRINTED['A.slopes']['interaction_F']} | F={s['interaction']['F']:.2f} p={s['interaction']['p']:.3f} | |")
        p = A.get("predictor")
        if p: rows.append(f"| predictor | ΔR2, p | {R.PRINTED['A.predictor']['delta_r2']}, {R.PRINTED['A.predictor']['p']} | {p['delta_r2']:.3f}, {('%.3f' % p['p']) if p.get('p') is not None else '--'} | R2 single {p['r2_single']} |")
        for b, v in R.PRINTED["A.alpha"].items():
            a = A["alpha"].get(b); rows.append(f"| tab:alpha-fits | {b} | {v} | {a['alpha']['mean']:.2f} | {a['ci_lo']['mean']:.2f}..{a['ci_hi']['mean']:.2f} |" if a else f"| tab:alpha-fits | {b} | {v} | -- | |")
        for b, e in A.get("ecf", {}).items():
            fc, fu = e["arms"]["clipped"]["fit"].get("slope"), e["arms"]["unclipped"]["fit"].get("slope")
            rows.append(f"| tab:ecf-rates | {b} | clipped {R.PRINTED['A.ecf_slopes']['clipped']}, unclipped {R.PRINTED['A.ecf_slopes']['unclipped']} | "
                        f"clipped {('%.3f' % fc) if fc is not None else '--'}, unclipped {('%.3f' % fu) if fu is not None else '--'} | |")
    return "\n".join(rows)


def render_all(A: Optional[dict] = None) -> Dict[str, str]:
    """Every emitted file as text, keyed by file name (the single source for write_all and check_all)."""
    out: Dict[str, str] = {}
    if A:
        for name, fn in (("A_tab_gamma_min", tab_gamma_min), ("A_tab_random_anchor", tab_random_anchor), ("A_tab_residual_check", tab_residual_check),
                         ("A_tab_slope_fits", tab_slope_fits), ("A_tab_ecf_rates", tab_ecf_rates), ("A_tab_alpha_fits", tab_alpha_fits),
                         ("A_tab_method_comparison", tab_method_comparison), ("A_predictor", predictor_text)):
            out[f"{name}.tex"] = fn(A)
    out["DIFF.md"] = diff_report(A)
    return out


def write_all(out_dir: Path, A: Optional[dict] = None) -> List[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); written = []
    for name, text in render_all(A).items():
        p = out_dir / name; p.write_text(text, encoding="utf-8"); written.append(p)
    return written


def check_all(out_dir: Path, A: Optional[dict] = None) -> Dict[str, str]:
    """Freeze check: regenerate every table in memory and compare with the files on disk, byte for byte.
    Returns {file: "same" | "differs" | "missing"}; a frozen repository has every value "same"."""
    out_dir = Path(out_dir); status = {}
    for name, text in render_all(A).items():
        p = out_dir / name
        status[name] = "missing" if not p.exists() else ("same" if p.read_text() == text else "differs")
    return status
