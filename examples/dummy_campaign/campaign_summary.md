# Dummy campaign summary

Campaign root: `examples/dummy_campaign`

## Top-level overview (steady-state, 50-90 ms after burst)

| FST | Diagnostic | Rate (kHz) | p0 (psia) | T0 (K) | Re/m x 10^-6 (1/m) | p0j (psia) | T0j (K) | p0j/pinf | pj/pinf | J |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FST_1388 | pls | 100.000 | 415.6 | 300.4 | 39.10 | 88.8 | 300.0 | 1058.62 | 25.19 | 4.64 |
| FST_1391 | piv | 100.000 | 418.7 | 298.9 | 39.73 | 105.0 | 300.0 | 1242.62 | 29.57 | 5.45 |
| FST_1391 | shift | 10.000 | 418.7 | 298.9 | 39.73 | 105.0 | 300.0 | 1242.62 | 29.57 | 5.45 |

Notes:
- Top-level values are computed from each FST LVM in the 50-90 ms post-burst window.
- pinf is computed from p0 using isentropic relations with tunnel M=7.2.
- Re/m is computed from static freestream state derived from p0/T0 at tunnel M=7.2.
- Jet enabled: pj and J are computed using isentropic relations with jet M=3.09.
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
- T0j assumed constant at 300 K.
- pinf computed from p0 using isentropic relation at M=7.2.
- Reference pinf at jet Mach from p0 uses M=3.09.
- pj computed from p0j using isentropic relation at jet M=3.09.

### Diagnostics

| Diagnostic | Known | Runs |
| --- | :---: | ---: |
| pls | yes | 2 |

### Runs

| Diagnostic | Run ID | Support run | Inferred rate (Hz) | Notes | Errors |
| --- | --- | :---: | ---: | --- | --- |
| pls | run_S0001 | no | 100000.000 |  |  |
| pls | scale_S0001 | yes | 50.000 | Support run (scale/cal); not treated as primary flow data. |  |

## FST_1391

### LVM condition notes

- Steady-state window: 50-90 ms after burst (indices 738-938).
- T0j assumed constant at 300 K.
- pinf computed from p0 using isentropic relation at M=7.2.
- Reference pinf at jet Mach from p0 uses M=3.09.
- pj computed from p0j using isentropic relation at jet M=3.09.

### Diagnostics

| Diagnostic | Known | Runs |
| --- | :---: | ---: |
| piv | yes | 3 |
| shift | no | 1 |

### Runs

| Diagnostic | Run ID | Support run | Inferred rate (Hz) | Notes | Errors |
| --- | --- | :---: | ---: | --- | --- |
| piv | cal_S0001 | yes | 500.000 | Support run (scale/cal); not treated as primary flow data. |  |
| piv | run_S0001 | no | 100000.000 |  |  |
| piv | scale_S0001 | yes | 500.000 | Support run (scale/cal); not treated as primary flow data. |  |
| shift | shift_S0001 | no | 10000.000 |  |  |
