"""pipeline -- the real-data replication pipeline for the Mahalanobis-margin paper
(NeurIPS 2026, submission 4828). This repository releases the paper-A code; the
companion paper's code is released separately at
github.com/miladkhademinori/structural-bias.

Design
------
Every number printed in the paper that depends on a real backbone is produced
by one of two stages:

1. ``features``  (GPU, Colab): one forward pass per (backbone, dataset, split),
   cached as ``features/<backbone>/<dataset>/<split>.npz``.  Extraction is a
   fixed-order forward pass, so caches are seed-independent and idempotent.
2. ``analyze``   (CPU, anywhere): every head, statistic, fit and table is computed
   from the caches with numpy/scipy only, seeded by (SEED, purpose) strings.

The only experiment that needs a GPU beyond feature extraction is the drift
study (``pipeline.drift``, fine-tunes the backbone).

Modules
-------
registry   backbones, datasets, task splits, seeds, paper-printed context values
features   Colab-side extraction (torch/transformers), cache format, synthetic caches
stats      class means, Ledoit-Wolf pooled covariance, gamma/D/kappa/nu, bias direction
heads      linear heads (within-task, joint, diagonal-joint), AdamW/L-BFGS/clipped SGD,
           post-hoc de-centering, LDA/SLDA, NCM, FeCAM-style, RanPAC-style
paper_a    gamma table, anchors, Condition (S), TC gap, slopes, predictor regression,
           E_CF sweep, Hill alpha, Algorithm 1 ("ours") and the frozen-feature baselines
drift      drift trajectories (GPU)
tables     LaTeX emitters and the paper-vs-reproduced diff
"""

__version__ = "2026.10.22"
