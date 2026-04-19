import cv2
import numpy as np

# ==========================================
# --- CONFIGURATION PARAMETERS ---
# ==========================================
VIDEO_PATH          = "video1.mp4"
FRAME_SCALE         = 0.5    # Downscales for massive performance boost
MAX_DIV_EXPECTED    = 2.0    # Absolute max divergence to stop heatmap flashing
DIV_THRESHOLD       = 0.4    # Minimum divergence value to track
EMA_ALPHA           = 0.5    # Tracking speed (0.1 = slow/smooth, 1.0 = instant)
BORDER_PCT          = 0.06   # Percentage of screen edges to ignore
TOP_REGION_PCT      = 0.80   # Percentage of peak divergence to track

# ==========================================
# --- INITIALIZATION ---
# ==========================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read the first frame.")

# Resize first frame to match our main loop
first_frame = cv2.resize(first_frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
h, w = first_frame.shape[:2]
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Start tracking point in the center of the screen
smooth_x, smooth_y = w // 2, h // 2

# Pre-compute the edge boundary mask so we don't track off-screen glitches
BORDER = int(min(h, w) * BORDER_PCT)
border_mask = np.zeros((h, w), dtype=np.uint8)
border_mask[BORDER:h-BORDER, BORDER:w-BORDER] = 255

print("Playing Pinpoint Max-Div Tracker... Press 'q' to quit.")

# ==========================================
# --- MAIN LOOP ---
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video finished!")
        break

    # Downscale for real-time performance
    frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Calculate Dense Optical Flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=3, winsize=21,
        iterations=3, poly_n=7, poly_sigma=1.5, flags=0
    )

    # 2. Smooth the flow to remove pixel noise
    flow_smoothed = cv2.GaussianBlur(flow, (15, 15), 0)
    u = flow_smoothed[..., 0]
    v = flow_smoothed[..., 1]

    # --- THE MAGIC BULLET: Magnitude Weighting ---
    # Calculate how fast pixels are physically moving
    magnitude = cv2.magnitude(u, v)
    # Normalize speed from 0.0 to 1.0
    mag_norm = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

    # 3. Calculate Divergence (The expansion of the pixels)
    du_dx = np.gradient(u, axis=1)
    dv_dy = np.gradient(v, axis=0)
    divergence = du_dx + dv_dy

    # Multiply divergence by the raw speed to kill the "Ghost Car" wake
    weighted_divergence = divergence * mag_norm

    # Smooth the final divergence map
    blur_k = max(1, (min(h, w) // 20) | 1)
    divergence_smoothed = cv2.GaussianBlur(weighted_divergence, (blur_k, blur_k), 0)

    # 4. Fixed Heatmap Visualization
    # Clip map to only show positive divergence (things getting closer)
    div_pos = np.clip(divergence_smoothed, 0, MAX_DIV_EXPECTED)
    div_display = np.uint8((div_pos / MAX_DIV_EXPECTED) * 255)
    
    # Apply color and turn low-noise areas completely black
    heatmap = cv2.applyColorMap(div_display, cv2.COLORMAP_JET)
    heatmap[div_pos < DIV_THRESHOLD * 0.3] = 0  

    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    # 5. Target Detection (Region Centroid)
    # Use the border_mask to ignore the extreme edges of the screen
    _, max_val, _, _ = cv2.minMaxLoc(divergence_smoothed, mask=border_mask)

    if max_val > DIV_THRESHOLD:
        # Isolate the exact region where the divergence is highest
        high_div_mask = np.uint8(divergence_smoothed > max_val * TOP_REGION_PCT) * 255
        high_div_mask = cv2.bitwise_and(high_div_mask, border_mask)

        # Calculate the Center of Mass of that region
        M = cv2.moments(high_div_mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Fast Exponential Moving Average (EMA) for smooth tracking
            smooth_x = int(EMA_ALPHA * cx + (1 - EMA_ALPHA) * smooth_x)
            smooth_y = int(EMA_ALPHA * cy + (1 - EMA_ALPHA) * smooth_y)

        # Draw the tracking reticle
        r = max(15, min(h, w) // 30)
        cv2.circle(overlay, (smooth_x, smooth_y), r, (0, 0, 255), 3)
        cv2.circle(overlay, (smooth_x, smooth_y), 3, (0, 255, 0), -1)
        cv2.putText(overlay, f"APPROACH  val={max_val:.2f}",
                    (smooth_x - 70, smooth_y - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    else:
        # Robot is stationary / no incoming objects
        cv2.putText(overlay, "No significant approach",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)

    # Show the final composite image
    cv2.imshow('Robot Max-Div Alignment', overlay)
    
    # Store current frame for the next flow calculation
    prev_gray = gray

    # Press 'q' to exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()