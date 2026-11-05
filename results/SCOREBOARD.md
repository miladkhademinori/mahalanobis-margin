# Synthetic verification scoreboard

65 assertions from 9 suite records: 61 PASS, 2 MECHANISM-FAIL, 0 CALIBRATION-EXCURSION, 2 PAPER-QUESTION.
Claims (the verdict unit): 52 claims: 48 PASS, 2 MECHANISM-FAIL, 0 CALIBRATION-EXCURSION, 2 PAPER-QUESTION.

Vocabulary (docs/VERDICTS.md): MECHANISM-FAIL, the mechanism the claim asserts is not exhibited (an exact arm misses its numerical tolerance, a finite-sample arm misses by more than the battery's family-wise bound, or the spec design cannot exhibit the claim at any n, marked [design] with the variant arm printed beside it); CALIBRATION-EXCURSION, the mechanism is exhibited and one gate number is missed within the battery's bound, size in standard errors reported; PAPER-QUESTION, a deterministic finding about a printed paper number, carried to the owner's Q-list with the measured value.

## Claims

| paper | claim | arms | verdict |
|---|---|---|---|
| A | A23a | A23a (): PAPER-QUESTION | **PAPER-QUESTION** |
| A | A26 | A26 (): PAPER-QUESTION | **PAPER-QUESTION** |
| A | EX3.4 | EX3.4 (EXACT): MECHANISM-FAIL | **MECHANISM-FAIL** |
| A | EX3.7 | EX3.7 (EXACT): MECHANISM-FAIL | **MECHANISM-FAIL** |

PASS: A15, A16, A17, A18, A19, A20, A21, A22, A23b, A23c1, A23c2, A23c3, A23d, A24, A25, A27, A28, A29, A30, EX1.1, EX1.2, EX3.1, EX3.2, EX3.3, EX3.5, EX3.6, EX3.8, EX4.1, EX4.2, EX5.1, EX5.2, EX5.3, EX5.4, G1.1, G1.2, G1.3, G2, G3.1, G3.2, G3.3, G3.4, S4.1, S4.2, S4.3, S4.4, S4.5, S4.6, TRK1

Tier 1 (mechanism: algebraic and population identities, exact in all inputs) must be all PASS for the code to be trusted.
Tier 2 (calibration: Monte-Carlo and finite-sample estimates, including exact functionals of finite-sample heads) lists excursions with the suites' own diagnosis; they are re-gated by standard error (tests/zbattery.py and core.gates), never by widening tolerances.

## Tier 1: mechanism

16/18 PASS

| run | id | arm | verdict | value | threshold |
|---|---|---|---|---|---|
| runs\A\synthetic_margin\A_synthetic_margin_EX3.json | EX3.4 | EXACT | **MECHANISM-FAIL** | 8/9 nu-rows violate; violating nus [0.13, 0.23, 0.42, 0.49, 0.51, np.float64(0.5 | 0 violations |
| runs\A\synthetic_margin\A_synthetic_margin_EX3.json | EX3.7 | EXACT | **MECHANISM-FAIL** | 36 doubling pairs; see EX3.8 for the corrected exponent bracket | within bracket |

PASS: A30, A24, G1.3, EX1.1, EX1.2, EX3.1, EX3.2, EX3.3, EX3.5, EX3.6, EX3.8, EX4.1, EX5.1, EX5.2, EX5.3, EX5.4

## Tier 2: calibration

45/47 PASS

| run | id | arm | verdict | value | threshold |
|---|---|---|---|---|---|
| runs\A\drift\A_drift_EX10.json | A26 |  | **PAPER-QUESTION** | None | None |
| runs\A\drift\A_drift_EX9.json | A23a |  | **PAPER-QUESTION** | None | None |

PASS: A27, A28, A29, A23b, A23c1, A23c2, A23c3, A23d, A25, TRK1, G1.1, G1.2, G2[planted=-0.3000], G2[planted=-0.1250], G2[planted=-0.0625], G2[planted=-0.0890], G2[trip=-1/16], G2[trip=-1/8], G3.1, G3.2, G3.3, G3.4, S4.1, S4.2, S4.3, S4.4, S4.5, S4.6, A15[a=1.32], A15[a=1.45], A15[a=2.0], A16, A17[a=1.32], A18[a=1.32], A17[a=1.45], A18[a=1.45], A17[a=2.0], A18[a=2.0], A19[a=1.32], A19[a=1.45], A19[a=2.0], A20, A21, A22, EX4.2

## Diagnoses recorded by the suites for the excursions and paper questions

- runs\A\drift\A_drift_EX10.json A26: FINDING (flag to owner): the 3-point Pearson r on the printed slope set {0.164, 1.73, 8.20} is 0.9931 < 0.995; the paper's 0.998 is not recoverable from the printed table values. Even injecting the paper's own per-seed dispersions (0.012/0.14/0.62, 5 seeds/arm) lowers the expected 15-point r further (~0.988, within-level variance adds to the residual). Either the paper's 0.998 was computed on a different point set (e.g. per-stage slopes) or it is a rounding artifact.
- runs\A\drift\A_drift_EX9.json A23a: FINDING (flag to owner): G07's arm-1 cubic reproduces its crossing (t*=14.1953 vs 14.2) and floor (19.443 vs 19.5) but its own claimed OLS over stages 1-10 of -0.164 is arithmetically -0.2191. The -0.164 is recovered from a stages-1-5 pre-acceleration window (-0.1620, inside the band) -- the same pre-floor-window reading G07 itself flagged for arm 2. A cubic refit to BOTH -0.164@1..10 and t*=14.2 is 8.30 - 0.023539 t - 0.001333 t^3 but floors at 18.08 (vs 19.5): the three anchors are overdetermined for a 2-parameter cubic through gamma0=8.30.
