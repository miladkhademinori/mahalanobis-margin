# Printed vs reproduced

| table | cell | printed | reproduced | note |
|---|---|---|---|---|
| tab:gamma-min | vit_in21k/cifar100 | 8.3 ± 0.4 | $3.2 \pm 0.4$ | eq:gamma per-task-cov variant $2.8 \pm 0.3$ |
| tab:gamma-min | vit_in21k/imagenet_r | 5.2 ± 0.3 | $1.5 \pm 0.2$ | eq:gamma per-task-cov variant $1.4 \pm 0.1$ |
| tab:gamma-min | vit_in21k/cub200 | 12.1 ± 0.7 | -- | eq:gamma per-task-cov variant -- |
| tab:gamma-min | vit_in21k/omnibenchmark | 9.6 ± 0.5 | -- | eq:gamma per-task-cov variant -- |
| tab:gamma-min | dinov2_b14/cifar100 | 47.2 ± 1.2 | $10.8 \pm 0.9$ | eq:gamma per-task-cov variant $10.2 \pm 0.9$ |
| tab:gamma-min | dinov2_b14/imagenet_r | 39.1 ± 1.0 | $6.8 \pm 0.8$ | eq:gamma per-task-cov variant $6.7 \pm 0.8$ |
| tab:gamma-min | dinov2_b14/cub200 | 51.8 ± 1.4 | -- | eq:gamma per-task-cov variant -- |
| tab:gamma-min | dinov2_b14/omnibenchmark | 44.6 ± 1.1 | -- | eq:gamma per-task-cov variant -- |
| tab:gamma-min | clip_b16/cifar100 | 23.1 ± 0.8 | $4.9 \pm 0.3$ | eq:gamma per-task-cov variant $4.3 \pm 0.1$ |
| tab:gamma-min | clip_b16/imagenet_r | 18.9 ± 0.6 | $2.5 \pm 0.5$ | eq:gamma per-task-cov variant $2.4 \pm 0.4$ |
| tab:gamma-min | clip_b16/cub200 | 31.4 ± 0.9 | -- | eq:gamma per-task-cov variant -- |
| tab:gamma-min | clip_b16/omnibenchmark | 26.3 ± 0.7 | -- | eq:gamma per-task-cov variant -- |
| tab:random-anchor | cifar100 | γ 0.31, TC 0.189 | γ $0.15 \pm 0.02$, TC $-0.113 \pm 0.003$ | |
| tab:random-anchor | imagenet_r | γ 0.28, TC 0.211 | γ $0.07 \pm 0.00$, TC $-0.148 \pm 0.001$ | |
| tab:random-anchor | cub200 | γ 0.58, TC 0.159 | -- | |
| tab:random-anchor | omnibenchmark | γ 0.42, TC 0.174 | -- | |
| slopes | vit_in21k | -0.141 | -0.061 | R2 1.00 |
| slopes | dinov2_b14 | -0.099 | -0.005 | R2 1.00 |
| slopes | clip_b16 | -0.118 | -0.041 | R2 1.00 |
| slopes | pooled 12 | (-0.115, -0.128, -0.102) | -0.020 [-0.029602043274193645, -0.013491937028574668] | |
| slopes | pooled 16 | (-0.119, -0.134, -0.105) | -0.020 [-0.029602043274193645, -0.013491937028574668] | |
| predictor | ΔR2, p | 0.06, 0.042 | 0.000, 0.448 | R2 single {'gamma': 0.48113642641771937, 'probe': 0.9998465424722456, 'ncm': 0.9959216849368531} |
| tab:alpha-fits | vit_in21k | 1.32 | 40.52 | 33.18..50.65 |
| tab:alpha-fits | dinov2_b14 | 1.28 | 28.57 | 23.57..35.79 |
| tab:alpha-fits | clip_b16 | 1.35 | 34.08 | 27.93..43.25 |
| tab:ecf-rates | vit_in21k | clipped -0.31, unclipped -0.1 | clipped --, unclipped -- | |
| tab:ecf-rates | dinov2_b14 | clipped -0.31, unclipped -0.1 | clipped --, unclipped -- | |