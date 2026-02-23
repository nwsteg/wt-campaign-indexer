# Dummy campaign summary

Campaign root: `examples/dummy_campaign`

## Top-level overview (steady-state, 50-90 ms after burst)

| FST | Diagnostic | Rate (kHz) | p0 (psia) | T0 (K) | Re_1 x 10^-6 (1/m) | p0j (psia) | T0j (K) | p0j/p0 | p0j/pinf | J |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FST_1388 | pls |  | 415.6 | 296.9 | 634.31 | 299.2 | 298.1 | 0.72 | 20.4 | 0.72 |
| FST_1391 | piv |  | 418.7 | 295.6 | 642.47 | 300.4 | 296.1 | 0.72 | 20.4 | 0.72 |
| FST_1391 | shift |  | 418.7 | 295.6 | 642.47 | 300.4 | 296.1 | 0.72 | 20.4 | 0.72 |

Notes:
- Top-level values are computed from each FST LVM in the 50-90 ms post-burst window.
- Runs containing `scale` or `cal` in their IDs are marked as support runs in detailed sections and excluded when picking a primary rate for the overview table.

FST count: **2**

## FST overview

| FST | LVM | Diagnostic count | Run count |
| --- | --- | ---: | ---: |
| FST_1388 | FST_1388.lvm | 1 | 2 |
| FST_1391 | FST_1391.lvm | 2 | 4 |

## FST_1388

### LVM condition notes

- Steady-state window: 50-90 ms after burst (indices 744-944).

### Diagnostics

| Diagnostic | Known | Runs |
| --- | :---: | ---: |
| pls | yes | 2 |

### Runs

| Diagnostic | Run ID | Support run | Inferred rate (Hz) | Notes | Errors |
| --- | --- | :---: | ---: | --- | --- |
| pls | run_S0001 | no |  | Unable to infer frame rate from .cihx metadata. |  |
| pls | scale_S0001 | yes |  | Unable to infer frame rate from .cihx metadata.; Support run (scale/cal); not treated as primary flow data. |  |

## FST_1391

### LVM condition notes

- Steady-state window: 50-90 ms after burst (indices 738-938).

### Diagnostics

| Diagnostic | Known | Runs |
| --- | :---: | ---: |
| piv | yes | 3 |
| shift | no | 1 |

### Runs

| Diagnostic | Run ID | Support run | Inferred rate (Hz) | Notes | Errors |
| --- | --- | :---: | ---: | --- | --- |
| piv | cal_S0001 | yes |  | Unable to infer frame rate from .cihx metadata.; Support run (scale/cal); not treated as primary flow data. |  |
| piv | run_S0001 | no |  | Unable to infer frame rate from .cihx metadata. |  |
| piv | scale_S0001 | yes |  | Unable to infer frame rate from .cihx metadata.; Support run (scale/cal); not treated as primary flow data. |  |
| shift | shift_S0001 | no |  | Unable to infer frame rate from .cihx metadata. |  |
