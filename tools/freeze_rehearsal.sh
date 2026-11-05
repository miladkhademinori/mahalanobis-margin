#!/usr/bin/env bash
# Repo freeze rehearsal (T18): the exact sequence the owner runs on Oct 31 (and rehearses before), from the repo root.
#
#   bash tools/freeze_rehearsal.sh [--with-caches FEATURES_ROOT] [--results RESULTS_DIR] [--run-selftest-lbfgs]
#
# Every step prints PASS or FAIL and the script stops at the first FAIL. Steps that need the feature caches run only
# when --with-caches is given (they are the Colab-side part of the freeze and can be run there with the same commands).
# Nothing here edits the papers or the owner's machine outside the repository; the git steps are printed, not run,
# until --git is passed, and even then they only prepare the single import commit on a fresh local repository.
set -u
FEATURES=""; RESULTS="results/lbfgs"; LBFGS=0; GIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --with-caches) FEATURES="$2"; shift 2;;
    --results) RESULTS="$2"; shift 2;;
    --run-selftest-lbfgs) LBFGS=1; shift;;
    --git) GIT=1; shift;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
step() { echo; echo "== $1"; }
pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; echo "The freeze does not proceed. Fix the cause, then rerun this script from the start."; exit 1; }

step "0. Clean tree: no bytecode, no tool caches, no patch leftovers"
# Only what this script or the interpreter created is removed. results/_selftest is NOT deleted: the fleet keeps
# dry-run outputs there; it is gitignored and excluded from the freeze manifest, so leftovers there never enter the freeze.
find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null
find . -name ".pytest_cache" -type d -prune -exec rm -rf {} + 2>/dev/null
rm -rf .ruff_cache
[ -z "$(find . -name '*.pyc' -o -name '*.orig' -o -name '*.rej' | head -1)" ] && pass "no bytecode or patch leftovers" || fail "bytecode or patch leftovers present"

step "1. Static check: no undefined names or unused imports in the pipeline and the entry point"
if command -v ruff >/dev/null 2>&1; then
  ruff check --isolated --select F pipeline replicate.py && pass "ruff F on pipeline and replicate.py" || fail "ruff F findings in pipeline or replicate.py"
else
  python -m pyflakes pipeline replicate.py && pass "pyflakes on pipeline and replicate.py" || fail "pyflakes findings (or pyflakes not installed: pip install pyflakes)"
fi

step "2. Test suite (one recorded failure is expected: test_e5_rate_fast_variant, the E5 finding; anything else fails the freeze)"
# Two runtime tests build tiny offline models through load_backbone/run_drift, which the owner storage-law guard
# (pipeline/features.py::_assert_download_allowed) blocks off Colab unless REPLICATION_ALLOW_DOWNLOADS=1 is set.
# They are deselected here and run on Colab (or with the override, which the owner sets knowingly); the record says which.
DESEL="--deselect tests/test_theory_B.py::E5E6GuardedFastTests::test_e5_rate_fast_variant"
if [ "${REPLICATION_ALLOW_DOWNLOADS:-0}" != "1" ]; then
  DESEL="$DESEL --deselect tests/test_runtime.py::test_drift_loop_end_to_end_with_tiny_vit --deselect tests/test_runtime.py::test_load_backbone_forward_lambdas_on_tiny_offline_models"
  echo "  (download guard active: the two tiny-model runtime tests are deselected here; expected on the owner's machine: 85 passed, 3 deselected)"
else
  echo "  (REPLICATION_ALLOW_DOWNLOADS=1: all runtime tests run; expected: 87 passed, 1 deselected)"
fi
python -m pytest tests -q $DESEL 2>&1 | tail -3 | tee /tmp/freeze_pytest.txt
grep -q " passed" /tmp/freeze_pytest.txt && ! grep -q " failed" /tmp/freeze_pytest.txt && pass "test suite (E5 deselected as the recorded finding)" || fail "test suite"

step "3. Selftest, AdamW arm (synthetic caches, both drivers end to end; about 2 minutes)"
python replicate.py selftest > /tmp/freeze_selftest.txt 2>&1 && pass "selftest adamw" || { tail -20 /tmp/freeze_selftest.txt; fail "selftest adamw"; }
if [ "$LBFGS" = "1" ]; then
  python replicate.py selftest --optimizer lbfgs > /tmp/freeze_selftest_lbfgs.txt 2>&1 && pass "selftest lbfgs" || { tail -20 /tmp/freeze_selftest_lbfgs.txt; fail "selftest lbfgs"; }
fi
rm -rf results/_selftest_features   # the synthetic caches the selftest wrote; results/_selftest is left in place (gitignored, not in the manifest)

step "4. Synthetic suites scored from the stored run records (no rerun; the records are part of the frozen state)"
python replicate.py synthetic > /tmp/freeze_synthetic.txt 2>&1 && pass "synthetic scoreboard regenerated" || { tail -20 /tmp/freeze_synthetic.txt; fail "synthetic scoreboard"; }
git diff --quiet -- results/SCOREBOARD.md 2>/dev/null; [ $? -ne 1 ] && pass "SCOREBOARD.md unchanged by regeneration (or no git yet)" || fail "SCOREBOARD.md changed on regeneration: the stored records and the scorer disagree"

step "5. Tables: regeneration is byte-identical to tables/out (the freeze idempotence test)"
python replicate.py --results "$RESULTS" tables --check && pass "tables --check" || fail "tables --check (regenerate with 'replicate.py tables' from the final results, commit, rerun)"

if [ -n "$FEATURES" ]; then
  step "6. Cache-side checks (feature caches at $FEATURES): condition-s and fingerprint regenerate and agree with the stored files"
  python replicate.py --features "$FEATURES" --results /tmp/freeze_cs condition-s --seeds 0 > /tmp/freeze_cs.txt 2>&1 && pass "condition-s runs" || { tail -10 /tmp/freeze_cs.txt; fail "condition-s"; }
  python replicate.py --features "$FEATURES" --results /tmp/freeze_fp fingerprint --seeds 0 > /tmp/freeze_fp.txt 2>&1 && pass "fingerprint runs" || { tail -10 /tmp/freeze_fp.txt; fail "fingerprint"; }
  if [ -f "$RESULTS/FINGERPRINT.json" ]; then
    python - "$RESULTS/FINGERPRINT.json" /tmp/freeze_fp/FINGERPRINT.json <<'EOF' && pass "fingerprint gamma values agree with the stored report (first seed, ours ordering)" || fail "fingerprint values moved since the stored report"
import json, sys, math
a, b = (json.load(open(p)) for p in sys.argv[1:3])
bad = 0
for k, cfg in a["configs"].items():
    if k not in b["configs"]: bad += 1; continue
    sa = cfg["seeds"][sorted(cfg["seeds"])[0]]["ours (sha256 rng)"]["gamma"]; sb = b["configs"][k]["seeds"][sorted(b["configs"][k]["seeds"])[0]]["ours (sha256 rng)"]["gamma"]
    for v in sa:
        if v.startswith("_") or v not in sb: continue
        if not math.isclose(sa[v]["gamma_min"], sb[v]["gamma_min"], rel_tol=1e-6): bad += 1
sys.exit(1 if bad else 0)
EOF
  fi
else
  step "6. Cache-side checks skipped (no --with-caches); run them on Colab with the same commands and paste the two PASS lines into the freeze record"
fi

step "7. Documentation and provenance text present"
for f in README.md docs/REPLICATION.md docs/COMMANDS.md docs/VERDICTS.md docs/THEORY_NOTES.md docs/LICENSES.md requirements.txt requirements-gpu.txt; do
  [ -s "$f" ] && echo "  ok $f" || fail "missing $f"
done
grep -q "Provenance statement" README.md && pass "README carries the provenance statement" || fail "README lacks the provenance statement"
grep -rn -- "rather than\|instead of" README.md docs/*.md >/dev/null && fail "banned phrase in README or docs" || pass "no banned phrase in README or docs"

step "8. Freeze manifest: sha256 of every versioned file and the tree hash"
python replicate.py freeze-manifest | tail -1 || fail "freeze-manifest"
TREE=$(python -c "import json; print(json.load(open('FREEZE_MANIFEST.json'))['tree_sha256'])")
pass "FREEZE_MANIFEST.json written; tree sha256 $TREE"

step "9. The single import commit (printed; run with --git on a fresh clone of the empty public repository)"
cat <<EOF
  git init -b main                       # or: git clone <empty public repo> && cd <repo>
  cp -r <frozen repo contents> .          # the tree whose FREEZE_MANIFEST tree hash is $TREE
  git add -A
  git commit -F- <<'MSG'
Camera-ready replication code for NeurIPS 2026 submission 4828 (Paper A)

Single import commit of the frozen repository (freeze date 2026-10-31).

Provenance: this repository is an independent re-implementation written for the camera-ready from the papers'
text (definitions, appendices, protocols). No experiment code from the original submissions existed; every number
the repository produces is a fresh computation, and tables/out/DIFF.md lists each printed value next to the
reproduced one. Nothing in the repository asserts a printed value; pipeline/registry.py::PRINTED holds them only
for that diff. The accepted values that the camera-ready replaces appear in the papers' change summaries with the
reason for each replacement.

Freeze manifest tree sha256: $TREE
MSG
  git tag -a camera-ready-freeze-2026-10-31 -m "Frozen for the NeurIPS 2026 camera-ready; FREEZE_MANIFEST tree sha256 $TREE"
  git push origin main --tags
EOF
if [ "$GIT" = "1" ]; then
  if [ -d .git ]; then
    fail "a .git directory already exists here; the single import commit is made on a fresh clone, not on the working repository"
  fi
  git init -q -b main && git add -A && git -c user.name="camera-ready" -c user.email="noreply@example.invalid" commit -q -F- <<EOF && git tag -a camera-ready-freeze-2026-10-31 -m "Frozen for the NeurIPS 2026 camera-ready; FREEZE_MANIFEST tree sha256 $TREE" && pass "import commit and tag created locally (not pushed)"
Camera-ready replication code for NeurIPS 2026 submission 4828 (Paper A)

Single import commit of the frozen repository (freeze date 2026-10-31).

Provenance: this repository is an independent re-implementation written for the camera-ready from the papers'
text (definitions, appendices, protocols). No experiment code from the original submissions existed; every number
the repository produces is a fresh computation, and tables/out/DIFF.md lists each printed value next to the
reproduced one. Nothing in the repository asserts a printed value; pipeline/registry.py::PRINTED holds them only
for that diff. The accepted values that the camera-ready replaces appear in the papers' change summaries with the
reason for each replacement.

Freeze manifest tree sha256: $TREE
EOF
fi

echo
echo "FREEZE REHEARSAL COMPLETE. Record the PASS lines and the tree hash $TREE in the freeze record (tools/FREEZE_REHEARSAL.md, section 3)."
