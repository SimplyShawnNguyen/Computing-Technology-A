# Sprint 1 — Items 4 & 5 (Health Metrics)

**Owner:** Tola
**Scope:** Sprint-1 work items 4 and 5 (plus the unit tests from item 6, included so the metrics are verifiable).

## What's in here

| File | Purpose |
|---|---|
| `health_metrics.py` | Pure, camera-free health layer. Converts a `MaxDivProcessor.Result` into two cue-quality metrics. |
| `test_health_metrics.py` | 14 unit tests using synthetic `Result` objects (no camera, no RealSense needed). |

## The two metrics

1. **Vector count** — `result.n_vectors` (how many valid Lucas-Kanade flow vectors this frame; low = starved flow).
2. **Shear / divergence ratio** — `result.axis_strength / result.divmax_raw`, guarded so NaN / zero / negative / infinite denominators return `None` ("unavailable") instead of crashing or leaking `inf`.

## How to use

```python
import health_metrics as hm

metrics = hm.compute_health_metrics(result)   # result = MaxDivProcessor.Result

print(metrics.n_vectors)          # int
print(metrics.shear_div_ratio)    # float or None
print(metrics.shear_div_ratio_state)  # "available" / "unavailable"
```

Works with the **real** `Result` class or any duck-typed object with the same
field names (`n_vectors`, `axis_strength`, `divmax_raw`, `valid`, `tau_s`), so
it can be reused unchanged for replay mode.

## Run the tests

```bash
python -m unittest test_health_metrics -v
```

All 14 pass. Only stdlib needed (`unittest`, `math`, `types`, `dataclasses`).

## Notes for the team

- This module depends on **nothing** from the perception engine, so item 3's
  `HealthMetrics` structure can be merged into `health_metrics.HealthMetrics`
  (or vice-versa) when your code is ready.
- Good / marginal / bad thresholds are deliberately **not** final here — the
  sprint plan requires empirical values from Chris before classification.
  The `HealthThresholds` dataclass is the injection point for those later.
- Metric 2's validity gate (`divmax_raw > 1e-9`) mirrors the engine's own
  TTC-trust gate so the two stay consistent.
