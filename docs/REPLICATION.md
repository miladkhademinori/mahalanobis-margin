# How to regenerate the results store

Every family in `results/` regenerates from the code in this repository by one
of the `replicate.py` subcommands below; nothing in the store is hand-typed.
Feature caches live under `$REPLICATION_FEATURES` (default `./features`);
results under `./results`. The seed policy is deterministic (SHA-256 over
(seed, purpose, salt); H06/I01/I02 determinism records), so re-running a suite
reproduces its stored bytes.

## The store, family by family

| Family | What it holds | How to regenerate |
|---|---|---|
| `runs/A/synthetic_margin/` (EX1, EX3, EX4, EX5) | The closed-form margin-verification suites: one self-describing JSON envelope per run (configuration, seeds, `metrics.assertions` with each assertion's value, threshold and diagnosis; provenance `synthetic`; derived tables). | `python replicate.py synthetic --run` (re-runs every suite, then scores), or `python -m experiments.synthetic_margin --suites EX1,EX2,EX3,EX4,EX5` |
| `runs/A/falsification/` | The falsification battery (G1-G3): the boundary behaviour of the margin bound. | `python replicate.py synthetic --run`, or `python -m experiments.falsification --suites G1,G2,G3` |
| `runs/A/slope_fits/` | The log(TC gap) versus gamma_min regression suite (S4, EX6-EX8). | `python replicate.py synthetic --run`, or `python -m experiments.slope_fits --suites S4,EX6,EX7,EX8` |
| `runs/A/drift/` (EX9-EX11) | The synthetic drift suites (trajectory laws and horizon identities; ~3-8 min CPU, `--selfcheck` in seconds). Distinct from the real-feature drift study, which writes `paperA_drift.json` (GPU; `replicate.py drift`). | `python replicate.py synthetic --run`, or `python experiments/drift.py` |
| `runs/A/predictor_regression/` (CI1) | The R2/F predictor-regression audit. Provenance `printed_inputs_partial`: the printed input columns are required, and the record carries `INPUTS_MISSING` and never fabricates. | `python experiments/predictor_regression.py --inputs <rows.json> --out results/runs/A/predictor_regression/A_predictor_regression_CI1.json` |
| `INDEX.json` | The run registry: run id, path, provenance, suite status, created time, and the `ledger_close` annotations of the corrections ledger. | Updated by each suite writer at run time (same commands as above). |
| `SCOREBOARD.md` / `SCOREBOARD.json` | The claim-by-claim scoreboard over every suite record: 65 assertions / 52 claims with tiers and verdicts (vocabulary and summary: `docs/VERDICTS.md`). | `python replicate.py synthetic` (scores the stored records; `--run` re-runs the suites first) |
| `CORRECTIONS.md` | The corrections ledger: every correction applied after registration, named before/after/reason. Hand-kept: it records the annotations, the commands above regenerate the bytes. | edited per correction |
| `_selftest/`, `_selftest_features/` | The end-to-end selftest on synthetic caches (drivers and table emitters). Synthetic provenance; nothing here is a paper number. | `python replicate.py selftest` |
| `README.md` | The store guide. | hand-maintained |

## The real-feature families (feature caches required)

These commands produce the real-feature payloads the released tables read;
until the completing store push, the staged caches carry their provenance:

- `python replicate.py features [--backbones ...] [--datasets ...]` or
  `colab/01_features_paperA.ipynb` — the frozen-feature caches (GPU).
- `python replicate.py analyze --paper A [--seeds 0,1,2,3,4]` — heads, TC gaps
  and slopes into `results/paperA_results.json`, then the tables.
- `python replicate.py condition-s` — the Condition (S) clauses from the
  caches alone (`paperA_conditionS.json`).
- `python replicate.py fingerprint` — which estimator convention reproduces
  the printed gamma_min / nu / RHS (`FINGERPRINT.md` / `.json`; caches only,
  minutes on CPU).
- `python replicate.py features-timm --data-root ... --out-root features_timm`
  — the recovered timm protocol, into a separate cache root.
- `python replicate.py drift --data-root /content/data` — the real drift study
  (`paperA_drift.json`; GPU; resumable per (lr, seed)).
- `python replicate.py drift-readback` — P1..P8 / C1..C5 scored from
  `paperA_drift.json` into `DRIFT_READBACK.md`.
- `python replicate.py figures` — every figure, each with a `.data.json`
  sidecar.
- `python replicate.py audit-tex --paper A --tex <neurips_2026.tex>` — the
  printed-versus-regenerated ledger (`TEX_AUDIT_A.md` / `.json`).
- `python replicate.py tables [--check]` — the released table fragments and
  `tables/out/DIFF.md` from the store; `--check` is the freeze test (every
  released table regenerates byte for byte).

## Coverage matrix (the corrections-ledger closes)

Status of every registered run record, carrying the same rows as
`results/CORRECTIONS.md` (agent CI5, 2026-10-22):

| Record | Stored | Suite status | Coverage status |
|---|---|---|---|
| `A_drift_EX9` | final | FAIL (8 of 9 PASS) | closed by annotation (`ledger_close` in `INDEX.json`); FAIL A23a stands as an owner-flag with the measured values in the record (printed -0.164 recovered as a stages-1-5 window fit, -0.1620; the stated stages-1-10 OLS measures -0.2191); I04 exec-log narrative owed to the J wave |
| `A_drift_EX10` | final | FAIL (A27 PASS) | closed by annotation; FAIL A26 stands as an owner-flag (measured 3-point r = 0.9931 beside the printed 0.998); I04 owed |
| `A_drift_EX11` | final | PASS | current |
| `A_synthetic_margin_EX1` | final | PASS | current |
| `A_synthetic_margin_EX3` | final | FAIL (6 of 8 PASS) | closed by annotation; EX3.4 and EX3.7 stand as exact-arm findings with diagnoses in the record; the 1/8 and 1/16 tightness anchors (EX3.1, EX3.2) PASS; I03 owed |
| `A_synthetic_margin_EX4` | final | PASS | current |
| `A_synthetic_margin_EX5` | final | PASS | current |
| `A_slope_fits_synthetic` | final | PASS | current |
| `A_falsification_synthetic` | final | PASS | current |
| `A_predictor_regression_CI1` | recorded | INPUTS_MISSING | nulls printed as nulls; regenerates when the printed input columns are supplied |

No suite verdict was flipped, no measurement rewritten, no FAIL softened by
the closes; re-running the suites reproduces the stored bytes.
