
import argparse
import os
import sys

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Lucas–Kanade parameters
# ─────────────────────────────────────────────────────────────────────────────

LK_PARAMS = dict(
    winSize   = (21, 21),
    maxLevel  = 3,                   # pyramid levels → handles larger motions
    criteria  = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,                          # max iterations
        0.01                         # epsilon
    )
)

# Shi-Tomasi feature detection defaults
FEATURE_PARAMS = dict(
    maxCorners   = 300,
    qualityLevel = 0.01,
    minDistance  = 10,
    blockSize    = 7
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a dense heatmap from sparse (x, y, magnitude) points
# ─────────────────────────────────────────────────────────────────────────────

def sparse_to_heatmap(
    h: int,
    w: int,
    pts: np.ndarray,        # (N, 2) float32 – (x, y) positions
    mags: np.ndarray,       # (N,)   float32 – magnitude at each point
    sigma: float = 30.0
) -> np.ndarray:
    """
    Gaussian-splat each tracked point onto a 2-D accumulator, then
    normalise to [0, 255] uint8.

    Each point contributes a Gaussian blob of height = magnitude.
    This gives a smooth, dense heatmap from sparse LK tracks.
    """
    heat = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    # build a small Gaussian kernel
    ksize = int(6 * sigma) | 1          # odd, covers ±3σ
    k1d   = cv2.getGaussianKernel(ksize, sigma)
    kern  = (k1d @ k1d.T).astype(np.float32)
    hk    = ksize // 2

    for (x, y), m in zip(pts, mags):
        ix, iy = int(round(x)), int(round(y))

        # clip to image boundary
        x0, x1 = max(0, ix - hk), min(w, ix + hk + 1)
        y0, y1 = max(0, iy - hk), min(h, iy + hk + 1)

        kx0 = hk - (ix - x0)
        ky0 = hk - (iy - y0)
        kx1 = kx0 + (x1 - x0)
        ky1 = ky0 + (y1 - y0)

        heat  [y0:y1, x0:x1] += m   * kern[ky0:ky1, kx0:kx1]
        weight[y0:y1, x0:x1] +=       kern[ky0:ky1, kx0:kx1]

    # weighted average (avoid div-by-zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        heat = np.where(weight > 0, heat / weight, 0.0)

    # normalise to 0–255
    vmax = heat.max()
    if vmax > 0:
        heat = heat / vmax * 255.0

    return heat.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Main processing class
# ─────────────────────────────────────────────────────────────────────────────

class LKHeatmapPipeline:
    """
    Lucas–Kanade sparse optical flow → heatmap overlay pipeline.

    Parameters
    ----------
    max_pts    : maximum Shi-Tomasi corners per frame
    blur_sigma : Gaussian smoothing radius for the heatmap
    alpha      : blending weight of the heatmap overlay (0 = invisible, 1 = solid)
    colormap   : cv2 colormap constant (e.g. cv2.COLORMAP_JET)
    redetect_interval : re-run feature detection every N frames
    """

    def __init__(
        self,
        max_pts: int           = 300,
        blur_sigma: float      = 30.0,
        alpha: float           = 0.55,
        colormap: int          = cv2.COLORMAP_JET,
        redetect_interval: int = 15,
    ):
        self.max_pts            = max_pts
        self.blur_sigma         = blur_sigma
        self.alpha              = alpha
        self.colormap           = colormap
        self.redetect_interval  = redetect_interval

        # internal state
        self._prev_gray   = None
        self._prev_pts    = None
        self._frame_count = 0

        # stats collected for reporting
        self.flow_magnitudes = []        # mean magnitude per frame

    # ── feature detection ────────────────────────────────────────────────────

    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        params = dict(FEATURE_PARAMS)
        params["maxCorners"] = self.max_pts
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **params)
        return pts  # shape (N,1,2) or None

    # ── process one frame ────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns the frame annotated with a heatmap overlay.
        For the very first frame there is nothing to compare against,
        so the raw frame is returned unchanged.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── initialise on first frame ─────────────────────────────────────
        if self._prev_gray is None or self._prev_pts is None:
            self._prev_gray  = gray.copy()
            self._prev_pts   = self._detect_features(gray)
            self._frame_count += 1
            return frame.copy()

        # ── re-detect periodically ────────────────────────────────────────
        if self._frame_count % self.redetect_interval == 0:
            self._prev_pts = self._detect_features(self._prev_gray)

        self._frame_count += 1

        # no usable points → fall back
        if self._prev_pts is None or len(self._prev_pts) == 0:
            self._prev_gray = gray.copy()
            return frame.copy()

        # ── Lucas–Kanade tracking ─────────────────────────────────────────
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            self._prev_pts,
            None,
            **LK_PARAMS
        )

        # keep only successfully tracked points
        if curr_pts is None or status is None:
            self._prev_gray = gray.copy()
            return frame.copy()

        ok        = status.ravel() == 1
        good_prev = self._prev_pts[ok].reshape(-1, 2)   # (N, 2)
        good_curr = curr_pts[ok].reshape(-1, 2)         # (N, 2)

        if len(good_curr) == 0:
            self._prev_gray = gray.copy()
            self._prev_pts  = self._detect_features(gray)
            return frame.copy()

        # ── flow magnitude ────────────────────────────────────────────────
        flow_vecs = good_curr - good_prev               # (N, 2)
        mags      = np.linalg.norm(flow_vecs, axis=1)  # (N,)
        self.flow_magnitudes.append(float(mags.mean()))

        # ── build heatmap ─────────────────────────────────────────────────
        heatmap_gray = sparse_to_heatmap(h, w, good_curr, mags, self.blur_sigma)
        heatmap_color = cv2.applyColorMap(heatmap_gray, self.colormap)

        # ── blend with original frame ─────────────────────────────────────
        # Only blend pixels where there is actual flow signal
        mask = (heatmap_gray > 5).astype(np.float32)[:, :, np.newaxis]
        overlay = (
            frame.astype(np.float32) * (1 - self.alpha * mask)
            + heatmap_color.astype(np.float32) * (self.alpha * mask)
        ).clip(0, 255).astype(np.uint8)

        # ── draw tracked feature points ───────────────────────────────────
        for (cx, cy) in good_curr.astype(int):
            cv2.circle(overlay, (cx, cy), 2, (255, 255, 255), -1)

        # ── HUD text ──────────────────────────────────────────────────────
        n_pts   = len(good_curr)
        mean_m  = mags.mean()
        cv2.putText(overlay, f"Lucas-Kanade Heatmap",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(overlay, f"Tracked pts: {n_pts}  |  Mean flow: {mean_m:.2f} px",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
        cv2.putText(overlay, f"Frame: {self._frame_count}",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # ── colourbar legend ──────────────────────────────────────────────
        overlay = self._draw_colorbar(overlay)

        # ── update state ──────────────────────────────────────────────────
        self._prev_gray = gray.copy()
        self._prev_pts  = good_curr.reshape(-1, 1, 2).astype(np.float32)

        return overlay

    # ── colour-bar ───────────────────────────────────────────────────────────

    def _draw_colorbar(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        bar_h, bar_w = 12, 150
        bar_x, bar_y = w - bar_w - 15, h - 35

        gradient = np.linspace(0, 255, bar_w, dtype=np.uint8).reshape(1, -1)
        gradient = np.tile(gradient, (bar_h, 1))
        bar_color = cv2.applyColorMap(gradient, self.colormap)

        frame[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w] = bar_color
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (255, 255, 255), 1)
        cv2.putText(frame, "Low", (bar_x, bar_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(frame, "High", (bar_x + bar_w - 28, bar_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(frame, "Flow magnitude",
                    (bar_x, bar_y + bar_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        return frame


# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline on a video source
# ─────────────────────────────────────────────────────────────────────────────

def run(
    source,                     # str path or int camera index
    pipeline: LKHeatmapPipeline,
    output_path: str = None,    # if None, only display — no file saved
):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source '{source}'", file=sys.stderr)
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay  = max(1, int(1000 / fps))   # ms per frame so playback is real-time

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"  Input  : {source}  ({width}×{height} @ {fps:.1f} fps, {total} frames)")
        print(f"  Output : {output_path}")
    else:
        print(f"  Input  : {source}  ({width}×{height} @ {fps:.1f} fps, {total} frames)")
        print(f"  Displaying live — press Q to quit")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = pipeline.process_frame(frame)

        if writer:
            writer.write(annotated)

        cv2.imshow("LK Heatmap  (press Q to quit)", annotated)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── print summary ─────────────────────────────────────────────────────
    if pipeline.flow_magnitudes:
        mags = pipeline.flow_magnitudes
        print(f"\n  Flow magnitude stats over {len(mags)} frames:")
        print(f"    mean  : {np.mean(mags):.3f} px")
        print(f"    max   : {np.max(mags):.3f} px")
        print(f"    min   : {np.min(mags):.3f} px")
        print(f"    std   : {np.std(mags):.3f} px")

    if output_path:
        print(f"  Saved → {output_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Lucas–Kanade optical flow heatmap visualisation"
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input",  type=str, help="Path to input video file")
    src.add_argument("--camera", type=int, help="Camera device index (e.g. 0)")

    p.add_argument("--output",     type=str, default=None,
                   help="(Optional) Save output to this mp4 file. If omitted, only displays.")
    p.add_argument("--max_pts",    type=int,   default=300,
                   help="Max feature points to track (default: 300)")
    p.add_argument("--blur_sigma", type=float, default=30.0,
                   help="Gaussian sigma for heatmap blur (default: 30)")
    p.add_argument("--alpha",      type=float, default=0.55,
                   help="Heatmap overlay transparency 0–1 (default: 0.55)")
    p.add_argument("--colormap",   type=str,   default="JET",
                   choices=["JET", "HOT", "TURBO", "INFERNO", "PLASMA", "MAGMA"],
                   help="Colour map (default: JET)")
    return p.parse_args()


COLORMAP_MAP = {
    "JET"    : cv2.COLORMAP_JET,
    "HOT"    : cv2.COLORMAP_HOT,
    "TURBO"  : cv2.COLORMAP_TURBO,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "PLASMA" : cv2.COLORMAP_PLASMA,
    "MAGMA"  : cv2.COLORMAP_MAGMA,
}


if __name__ == "__main__":
    args = parse_args()

    # determine source
    if args.camera is not None:
        source = args.camera
    elif args.input:
        source = args.input
    else:
        print("ERROR: Provide --input <video> or --camera <index>", file=sys.stderr)
        sys.exit(1)

    pipeline = LKHeatmapPipeline(
        max_pts    = args.max_pts,
        blur_sigma = args.blur_sigma,
        alpha      = args.alpha,
        colormap   = COLORMAP_MAP[args.colormap],
    )

    print(f"\nLucas–Kanade Heatmap Pipeline")
    print(f"  max_pts    = {args.max_pts}")
    print(f"  blur_sigma = {args.blur_sigma}")
    print(f"  alpha      = {args.alpha}")
    print(f"  colormap   = {args.colormap}\n")

    run(source, pipeline, output_path=args.output)
