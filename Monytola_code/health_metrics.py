#!/usr/bin/env python3
"""
health_metrics.py — UI-independent health layer for the cue-quality monitor.

SPRINT 1 SCOPE (work items 4 and 5)
------------------------------------
This module converts a single ``MaxDivProcessor.Result`` (the per-frame output
of the supplied perception engine) into the two cue-quality metrics the
operator panel will display:

    Metric 1 — Vector count
        The number of valid Lucas-Kanade flow vectors this frame
        (``result.n_vectors``). This is the primary "is there any flow to work
        with?" signal. A low count means the optical-flow estimate is
        *starved* (weak texture, fast motion, low light) and every other cue
        derived from it should not be trusted.

    Metric 2 — Shear / divergence ratio
        ``result.axis_strength / result.divmax_raw``. A scale-free "is this a
        real peak or just a slid ridge?" diagnostic: the shear magnitude
        relative to the isotropic divergence. It must be computed defensively
        because ``divmax_raw`` is frequently NaN (never computed), zero, or
        negative, and a blind division would produce NaN / ±Inf.

WHY A SEPARATE, PURE LAYER?
----------------------------
This module is deliberately free of any camera, OpenCV-window, or PyQt
dependency. The sprint plan requires the SAME metric logic to drive three
surfaces — the live operator panel, deterministic replay (no camera), and
the presenter view — and to be unit-testable with synthetic Result objects.
Keeping it pure means:

  * live mode feeds a real ``Result`` from ``RealSenseCue``;
  * replay mode feeds a ``Result`` reconstructed from a logged CSV;
  * tests feed a hand-built fake with no camera attached;

and all three get identical numbers.

The functions here only *read* attributes off the Result object (duck typing),
so they work with the real nested class OR any test double that exposes the
same field names. Nothing is imported from ``maxdiv_perception``, which keeps
this module cheap to import and easy to test in isolation.

Sprint 1 deliberately does NOT invent final good/marginal/bad thresholds:
those are Sprint-2 work that must be calibrated against the supervisor's
empirical data (sprint-plan "never cut for polish" rule). For now each metric
reports an "available / unavailable" state; the full classification
vocabulary is defined here so item 9 can extend it without reshaping the
data object.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Metric state vocabulary
# ---------------------------------------------------------------------------
class MetricState:
    """String constants for a metric's display state.

    Sprint 1 uses only UNAVAILABLE and AVAILABLE. GOOD / MARGINAL / BAD are
    reserved for sprint-plan item 9 (health states); they are declared now so
    callers can switch on them later without a rename.
    """

    UNAVAILABLE = "unavailable"
    AVAILABLE = "available"
    GOOD = "good"
    MARGINAL = "marginal"
    BAD = "bad"


# ---------------------------------------------------------------------------
# Provisional thresholds (item 9 — DO NOT treat as final)
# ---------------------------------------------------------------------------
# The sprint plan is explicit that final good/marginal/bad thresholds must
# come from the supervisor's empirical measurements, not from invention.
# `min_divergence` below mirrors the perception engine's own `divmax_raw > 1e-9`
# validity gate (see maxdiv_perception.py) so the two stay consistent.

@dataclass(frozen=True)
class HealthThresholds:
    """Configurable thresholds used by the metric helpers.

    Kept as a small frozen dataclass (rather than scattered module globals) so
    the operator panel can inject its own values at runtime and the defaults
    stay visible in one place.

    Attributes
    ----------
    min_vectors:
        Provisional minimum valid-vector count. Below this the flow field is
        considered "starved". (Sprint 1: informational only — the actual
        classification is item 9.)
    min_divergence:
        Divergence floor below which the shear/divergence ratio is
        meaningless. Defaults to the same 1e-9 gate the perception engine
        uses before it will publish a time-to-contact estimate.
    """

    min_vectors: int = 5
    min_divergence: float = 1e-9


DEFAULT_THRESHOLDS = HealthThresholds()


# ---------------------------------------------------------------------------
# HealthMetrics — the data object the UI consumes
# ---------------------------------------------------------------------------
@dataclass
class HealthMetrics:
    """Per-frame health snapshot.

    This is PURE DATA — no logic, no UI, no camera. It is the contract between
    the metrics layer and whatever renders it (operator panel, presenter view,
    or a future replay inspector).

    Fields
    ------
    timestamp : Optional[float]
        Wall-clock time of the frame in seconds, if the caller supplied one.
    frame_index : int
        Monotonic frame counter (useful for replay / plot x-axes).
    n_vectors : int
        METRIC 1 raw value — number of valid LK vectors this frame.
    n_vectors_state : str
        METRIC 1 state (Sprint 1: always "available"; see MetricState).
    shear_div_ratio : Optional[float]
        METRIC 2 raw value — ``axis_strength / divmax_raw``, or ``None`` when
        the ratio cannot be computed safely (see ``metric2_shear_divergence``).
    shear_div_ratio_state : str
        METRIC 2 state ("available" / "unavailable").
    arc_dir_spread_deg : Optional[float]
        METRIC 3 raw value — circular std-deviation of the E_m arc DIRECTION
        over the last few frames, in degrees (lower = more stable). ``None``
        when fewer than 2 usable arc vectors are available this window.
    arc_dir_stability : Optional[float]
        METRIC 3 raw value — the mean resultant length ``R`` in ``[0, 1]`` of
        the same window (``1.0`` = every frame agrees on one line, ``0.0`` =
        directions uniformly spread).
    arc_dir_spread_state : str
        METRIC 3 state ("available" / "unavailable").
    cue_valid : bool
        Pass-through of ``result.valid`` — whether the perception engine
        considers its own TTC estimate trustworthy this frame.
    tau_s : Optional[float]
        Pass-through of ``result.tau_s`` (time-to-contact in seconds) for the
        HUD. Not a health metric; carried along for convenience only.
    """

    timestamp: Optional[float] = None
    frame_index: int = 0

    n_vectors: int = 0
    n_vectors_state: str = MetricState.UNAVAILABLE

    shear_div_ratio: Optional[float] = None
    shear_div_ratio_state: str = MetricState.UNAVAILABLE

    arc_dir_spread_deg: Optional[float] = None
    arc_dir_stability: Optional[float] = None
    arc_dir_spread_state: str = MetricState.UNAVAILABLE

    cue_valid: bool = False
    tau_s: Optional[float] = None


# ---------------------------------------------------------------------------
# Metric 1 — valid-vector count
# ---------------------------------------------------------------------------
def metric1_vector_count(
    result: Any,
    thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
) -> Tuple[int, str]:
    """METRIC 1: read the number of valid optical-flow vectors.

    ``result.n_vectors`` is set by the perception engine on every processed
    frame and is always a non-negative integer, so this metric cannot be
    "missing" in the arithmetic sense. (Whether a *low* count means "bad" is
    the item-9 classification concern, not this function's.)

    Parameters
    ----------
    result : any object exposing ``n_vectors`` (duck-typed).
    thresholds : unused for this metric; accepted for a uniform call signature.

    Returns
    -------
    (value, state) — value is the raw count, state is "available".
    """
    # getattr with a default keeps this safe even if a partial/fake Result
    # omits the field during early development; int() normalises any float
    # or bool that might sneak in.
    value = int(getattr(result, "n_vectors", 0) or 0)
    return value, MetricState.AVAILABLE


# ---------------------------------------------------------------------------
# Metric 2 — shear / divergence ratio
# ---------------------------------------------------------------------------
def metric2_shear_divergence(
    result: Any,
    thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
) -> Tuple[Optional[float], str]:
    """METRIC 2: ``axis_strength / divmax_raw``, guarded against invalid input.

    The shear magnitude (``axis_strength``) is compared against the isotropic
    divergence (``divmax_raw``) as a scale-free cue-quality diagnostic. The
    division is guarded because in real operation ``divmax_raw`` is frequently:

      * NaN              — the field was never computed this frame,
      * 0.0              — degenerate (divide-by-zero), or
      * tiny / negative  — numerically meaningless for an approach cue.

    Guard rule: the ratio is returned only when ``divmax_raw`` is finite AND
    strictly greater than ``thresholds.min_divergence``. This mirrors the
    perception engine's own gate (``divmax_raw > 1e-9``) that decides whether
    ``tau_s`` is trustworthy, so the two stay consistent.

    Parameters
    ----------
    result : any object exposing ``axis_strength`` and ``divmax_raw``.
    thresholds : optional ``HealthThresholds`` override for the floor.

    Returns
    -------
    (ratio, state) — ``ratio`` is None and state "unavailable" when the
    denominator is invalid; otherwise the float division result and
    "available".
    """
    divmax_raw = getattr(result, "divmax_raw", float("nan"))
    axis_strength = getattr(result, "axis_strength", 0.0)

    # NaN and infinities fail the strict `<=` comparison, so `math.isfinite`
    # plus the floor check covers every bad-denominator case in one expression.
    if not math.isfinite(divmax_raw) or divmax_raw <= thresholds.min_divergence:
        return None, MetricState.UNAVAILABLE

    ratio = float(axis_strength) / float(divmax_raw)
    # A finite denominator can still produce inf if axis_strength overflows;
    # treat that as unavailable rather than leaking an infinity to the UI.
    if not math.isfinite(ratio):
        return None, MetricState.UNAVAILABLE

    return ratio, MetricState.AVAILABLE


# ---------------------------------------------------------------------------
# Metric 3 — arc-direction stability (Sprint 5, item 1)
# ---------------------------------------------------------------------------
def metric3_arc_direction_stability(
    arc_vecs: Sequence[Tuple[float, float]],
    thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
) -> Tuple[Optional[float], Optional[float], str]:
    """METRIC 3: stability of the E_m arc DIRECTION over a recent frame window.

    The E_m arc direction is a LINE — ``result.arc_vec`` is a signed unit
    vector, but the axis is only defined modulo 180°, so ``+v`` and ``-v`` are
    the SAME arc. To measure how much the direction wobbles across frames we
    must therefore treat anti-parallel vectors as identical.

    We do that by folding each vector into its DOUBLED-ANGLE representation
    ``phi = 2 * atan2(y, x)``, which maps a line to a single point on the
    circle (removing the 180° ambiguity), then computing standard circular
    statistics over those doubled angles:

        stability  = mean resultant length R in [0, 1]
                     (1 = every frame agrees on one line; 0 = uniformly spread)
        spread_deg = circular std-deviation of the line direction in degrees,
                     in [0, 90]  (lower = more stable)

    Because ``phi`` depends only on the line (never its sense), a vector and
    its own negation contribute the SAME phi and are treated as equal — that is
    the correct handling of the 180° direction ambiguity.

    Parameters
    ----------
    arc_vecs :
        A sequence of recent ``(ax, ay)`` signed arc unit vectors (e.g. the
        last ``window_size`` frames' ``result.arc_vec``). Degenerate
        (near-zero) and unreadable entries are skipped.
    thresholds :
        Accepted for a uniform call signature; not used by this metric yet
        (good/marginal/bad boundaries are later Sprint work).

    Returns
    -------
    (spread_deg, stability, state)
        spread_deg : Optional[float] — line-direction circular std dev (deg).
        stability : Optional[float] — mean resultant length R in [0, 1].
        state     : "available" when >= 2 usable vectors, else "unavailable".
    """
    phis: list[float] = []
    for vec in arc_vecs or ():
        try:
            x, y = float(vec[0]), float(vec[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.hypot(x, y) <= 1e-12:
            continue                       # no usable direction this frame
        phis.append(2.0 * math.atan2(y, x))

    if len(phis) < 2:
        return None, None, MetricState.UNAVAILABLE

    mc = sum(math.cos(p) for p in phis) / len(phis)
    ms = sum(math.sin(p) for p in phis) / len(phis)
    R = math.hypot(mc, ms)

    stability = float(min(max(R, 0.0), 1.0))

    if R <= 1e-12:
        # Uniformly spread around the full doubled circle -> maximally unstable.
        spread_deg = 90.0
    else:
        # circular std dev in the doubled space; halve back to line degrees,
        # clamped to the max a line can deviate (90°) from its mean.
        sigma_phi = math.sqrt(-2.0 * math.log(R))
        spread_deg = min(math.degrees(sigma_phi) / 2.0, 90.0)

    return round(spread_deg, 3), round(stability, 4), MetricState.AVAILABLE


class ArcDirectionStabilityWindow:
    """Rolling buffer of the last N E_m arc directions (Sprint 5, item 1).

    ``compute_health_metrics`` is deliberately pure and stateless, so the
    caller that drives frames owns the "last few frames" of history this metric
    needs. This class provides that state: feed it one ``result.arc_vec`` per
    frame (``window.update(result.arc_vec)``), then either ask it for the
    current spread/stability directly or hand ``window.arc_vecs()`` to
    ``compute_health_metrics(arc_history=...)``.

    A window of size 1 cannot produce a spread (the metric needs >= 2 samples),
    so ``window_size`` is clamped to a minimum of 2.

    Parameters
    ----------
    window_size : int
        How many recent arc directions to retain (default 5).
    """

    def __init__(self, window_size: int = 5):
        self.window_size = max(2, int(window_size))
        self._deque: collections.deque = collections.deque(maxlen=self.window_size)

    def __len__(self) -> int:
        return len(self._deque)

    def update(self, arc_vec: Any) -> None:
        """Push the latest ``(ax, ay)`` arc vector; skip ``None``/unreadable."""
        if arc_vec is None:
            return
        try:
            x, y = float(arc_vec[0]), float(arc_vec[1])
        except (TypeError, ValueError, IndexError):
            return
        self._deque.append((x, y))

    def arc_vecs(self) -> list:
        """Current window as a list of ``(ax, ay)`` tuples (oldest -> newest)."""
        return list(self._deque)

    def spread_and_stability(
        self, thresholds: HealthThresholds = DEFAULT_THRESHOLDS
    ) -> Tuple[Optional[float], Optional[float], str]:
        """Convenience wrapper: compute METRIC 3 over the current window."""
        return metric3_arc_direction_stability(self.arc_vecs(), thresholds)

    def reset(self) -> None:
        """Clear the window (e.g. on a scene cut / camera re-acquisition)."""
        self._deque.clear()


# ---------------------------------------------------------------------------
# Top-level pure function: Result -> HealthMetrics
# ---------------------------------------------------------------------------
def compute_health_metrics(
    result: Any,
    thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
    timestamp: Optional[float] = None,
    frame_index: int = 0,
    arc_history: Optional[Sequence[Tuple[float, float]]] = None,
) -> HealthMetrics:
    """Convert one perception ``Result`` into a ``HealthMetrics`` snapshot.

    This is the single entry point the UI calls each frame. It performs no I/O
    and no perception — it only reads fields off ``result`` — so it is
    trivially unit-testable with a synthetic Result (see test_health_metrics.py).

    Parameters
    ----------
    result :
        A ``MaxDivProcessor.Result`` (or any object exposing the same field
        names) from the supplied perception engine.
    thresholds :
        Optional ``HealthThresholds`` override for the classification floors.
    timestamp :
        Optional wall-clock time to stamp the snapshot with.
    frame_index :
        Monotonic frame number for replay / plot axes.
    arc_history :
        Optional list of recent ``(ax, ay)`` arc unit vectors (from an
        ``ArcDirectionStabilityWindow``) used to compute METRIC 3. When
        omitted, METRIC 3 reports "unavailable" — a single frame cannot
        produce a spread, which needs >= 2 samples.

    Returns
    -------
    A fully-populated ``HealthMetrics`` object.
    """
    n_vectors, n_vectors_state = metric1_vector_count(result, thresholds)
    ratio, ratio_state = metric2_shear_divergence(result, thresholds)

    # METRIC 3 — arc-direction stability over the caller-owned rolling window.
    # The pure layer stays stateless: if the caller owns a window (an
    # ArcDirectionStabilityWindow) it passes it here; otherwise we fall back to
    # a single-frame window from this result, which cannot yet yield a spread
    # (the metric needs >= 2 frames) and so reports unavailable.
    if arc_history is None:
        current = getattr(result, "arc_vec", None)
        window = [current] if current is not None else []
    else:
        window = list(arc_history)
    spread_deg, stability, arc_state = metric3_arc_direction_stability(window, thresholds)

    return HealthMetrics(
        timestamp=timestamp,
        frame_index=frame_index,
        n_vectors=n_vectors,
        n_vectors_state=n_vectors_state,
        shear_div_ratio=ratio,
        shear_div_ratio_state=ratio_state,
        arc_dir_spread_deg=spread_deg,
        arc_dir_stability=stability,
        arc_dir_spread_state=arc_state,
        # Convenience pass-throughs so the HUD can show "is the cue valid?"
        # and the current TTC without reaching back into the Result object.
        cue_valid=bool(getattr(result, "valid", False)),
        tau_s=getattr(result, "tau_s", None),
    )


# ---------------------------------------------------------------------------
# Self-test (no camera required)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # A minimal stand-in for a perception Result — the same duck-typed shape a
    # real MaxDivProcessor.Result exposes for these fields.
    from types import SimpleNamespace

    good = SimpleNamespace(
        n_vectors=18, axis_strength=0.02, divmax_raw=0.05, valid=True, tau_s=2.0,
        arc_vec=(1.0, 0.0),
    )
    starved = SimpleNamespace(
        n_vectors=0, axis_strength=0.0, divmax_raw=float("nan"), valid=False, tau_s=None,
        arc_vec=(1.0, 0.0),
    )

    print("Good frame    ->", compute_health_metrics(good, frame_index=1))
    print("Starved frame ->", compute_health_metrics(starved, frame_index=2))

    # METRIC 3 — a window that keeps to the SAME arc line (even flipping sign,
    # which is the SAME line) must be perfectly stable...
    stable = ArcDirectionStabilityWindow(5)
    for v in [(1.0, 0.0), (0.999, 0.001), (-1.0, 0.0), (-0.999, -0.001)]:
        stable.update(v)
    print("Stable line   ->", stable.spread_and_stability())

    # ...while a window that swings two frames to PERPENDICULAR lines is maximally spread.
    wild = ArcDirectionStabilityWindow(5)
    for v in [(1.0, 0.0), (0.0, 1.0)]:
        wild.update(v)
    print("Perpendicular->", wild.spread_and_stability())
