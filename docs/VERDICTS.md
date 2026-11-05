# Verdicts: what the verification suites measured

The verdict vocabulary of the scoreboard (`results/SCOREBOARD.md`, generated
by `experiments/scoreboard.py`) and the per-table verdict summary, stated in
the paper's own measurement voice: every number below is a stored measurement,
printed beside the paper's printed value wherever one exists.

## Vocabulary

- **PASS** — the assertion held: at machine precision in tier 1, at the
  battery's calibrated tolerance in tier 2.
- **MECHANISM-FAIL** — the mechanism the claim asserts is not exhibited: an
  exact arm misses its numerical tolerance, a finite-sample arm misses by more
  than the battery's family-wise bound, or the spec design cannot exhibit the
  claim at any n (marked [design], with the variant arm printed beside it).
- **CALIBRATION-EXCURSION** — the mechanism is exhibited and one gate number
  is missed within the battery's bound; the size of the miss is reported in
  standard errors. Excursions are re-gated by standard error
  (`tests/zbattery.py`, `core/gates.py`), never by widening tolerances.
- **PAPER-QUESTION** — a deterministic finding about a printed paper number,
  carried to the owner's Q-list with the measured value.

Tier 1 (mechanism: algebraic and population identities, exact in all inputs)
must be all PASS for the code to be trusted. Tier 2 (calibration: Monte-Carlo
and finite-sample estimates, including exact functionals of finite-sample
heads) lists excursions with the suites' own diagnoses.

## The scoreboard as measured (Paper A store)

65 assertions from 9 suite records: 61 PASS, 2 MECHANISM-FAIL,
0 CALIBRATION-EXCURSION, 2 PAPER-QUESTION. Claims (the verdict unit): 52
claims — 48 PASS, 2 MECHANISM-FAIL, 2 PAPER-QUESTION. Tier 1: 16/18 PASS.
Tier 2: 45/47 PASS.

## Verdicts by table

### tab:drift-slopes and fig:drift-trajectory (drift suites EX9-EX11, `runs/A/drift/`)

- EX11 PASS; EX9 8 of 9 PASS; EX10 A27 PASS (ratio 10.5488, inside [9, 11]).
- The horizon claims verify as measured: t* = 14.1953 / 2.341 / 0.2341 against
  the printed 14.2 / 2.35 / 0.21 (A23c1-c3 PASS); time-to-floor 0.474 against
  the printed 0.48 (A23d PASS); the secant identity exact to 1e-12 (A24 PASS);
  the t* ratio 6.06 in the sub-linear band (A25 PASS).
- A23a PAPER-QUESTION (owner-flag, full diagnosis in the run record): the
  printed -0.164 slope is recovered as a stages-1-5 pre-acceleration window
  fit (-0.1620, inside the band); the stated stages-1-10 OLS measures -0.2191.
- A26 PAPER-QUESTION (owner-flag, full diagnosis in the run record): the
  3-point Pearson r on the printed slope set {0.164, 1.73, 8.20} measures
  0.9931 beside the printed 0.998.

### The margin-bound tables (synthetic_margin suites, `runs/A/synthetic_margin/`)

- EX1 PASS (EX1.1, EX1.2); EX4 PASS (EX4.1, EX4.2); EX5 PASS (EX5.1-EX5.4).
  The corrections ledger records the letter row as "EX1, EX2, EX4, EX5 PASS".
- EX3: 6 of 8 sub-assertions PASS, including the 1/8 and 1/16 tightness
  anchors (EX3.1, EX3.2) and the corrected exponent bracket (EX3.8).
- EX3.4 MECHANISM-FAIL (exact arm, diagnosis stored): the interpolating
  exponent -(1-nu)/8 sits against the aligned -(1-nu/2)^2/8; the difference
  nu^2/32 exceeds the 0.005 tolerance for nu > 0.4.
- EX3.7 MECHANISM-FAIL (exact arm, diagnosis stored): the Gaussian-tail
  prefactor contributes ~2^-0.25 per doubling, beyond the 1.2 lower slack of
  the bracket.

### The falsification battery (G1-G3, `runs/A/falsification/`)

- G1.1, G1.2, G1.3 PASS; G2 PASS on all six planted/triplet arms; G3.1-G3.4
  PASS.

### The slope-fit tables (S4, `runs/A/slope_fits/`)

- S4.1-S4.6 PASS.

### The predictor regression (CI1, `runs/A/predictor_regression/`)

- Recorded INPUTS_MISSING (provenance `printed_inputs_partial`): the printed
  input columns are required; nulls are printed as nulls.

## Where the findings live

- Every verdict, value, threshold and diagnosis: the run records under
  `results/runs/A/` and the scoreboard `results/SCOREBOARD.md` / `.json`.
- The ledger closes and their wording: `results/CORRECTIONS.md` and the
  `ledger_close` fields of `results/INDEX.json`; the same status is carried in
  the coverage matrix of `docs/REPLICATION.md`.
- Printed-versus-regenerated table cells: `tables/out/DIFF.md` and
  `TEX_AUDIT_A.md` (from `python replicate.py audit-tex`).

The suite FAILs stand as findings about the printed numbers and tolerance
bands, each with the measured value beside the printed one; the two
PAPER-QUESTION flags are carried to the owner's Q-list with their measured
values, and nothing was flipped, rewritten or softened to produce this
summary.
