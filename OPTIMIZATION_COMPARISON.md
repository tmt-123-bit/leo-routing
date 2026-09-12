# Optimization Comparison

## Independent verification

| Scenario | Before | After | Change |
|---|---:|---:|---:|
| Medium load delivery rate | 78.1178% | 78.4102% | +0.2923 percentage points |
| Hotspot high load delivery rate | 29.4228% | 30.3270% | +0.9043 percentage points |

The comparison covers 16,400 independent episodes and 8,996,000 packet records in the original 24-satellite, 30-slot environment. Delivery, per-class delivery, other-destination delivery, successful-packet delay, hop count, and switching-rate protection checks all passed.

## What changed

- Medium load: two of eight policy seeds used updated Actor weights; the other six retained the original model after protected deployment selection.
- Hotspot high load: the gain came from the fixed direct-destination semantic input correction; the Actor weights were unchanged.
- The optimized package is opt-in and does not replace the historical default model automatically.

## Interpretation

The hotspot simultaneous lower bound was +0.6124 percentage points. The largest simultaneous switching-rate upper bound was 11.9849%, leaving about 0.0151 percentage points of margin below the 12% limit. The medium-load cross-seed interval includes zero, so the medium-load training contribution is descriptive rather than a universal guarantee.

The verified package and packet-level evidence remain in the local `outputs/verified-regime-model-20260911-v1/` and `outputs/model_optimization_evidence.md` records; generated model weights and large experiment logs are intentionally not part of the source submission.
