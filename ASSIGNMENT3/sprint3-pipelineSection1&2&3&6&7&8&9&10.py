import cv2
import numpy as np

# ================================
# Load video
# ================================
cap = cv2.VideoCapture("Foward_Fast(Textured).mp4")

ret, frame1 = cap.read()
ret, frame2 = cap.read()

if not ret or frame1 is None or frame2 is None:
    print("Error: Could not read video.")
    exit()

# ================================
# Section 10: Performance Optimisation (Van Tuy)
# ================================
# Process frames at a fixed smaller size to reduce CPU work.
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360
PROCESS_SIZE = (PROCESS_WIDTH, PROCESS_HEIGHT)

# Larger spacing make fewer vectors and faster Lucas-Kanade
skip = 45

# Farneback is dense and expensive, so calculate it on a smaller frame,
# then scale the result back to the display size.
farneback_scale = 0.5

# Draw fewer Farneback vectors because dense arrows are expensive and cluttered.
farneback_draw_step = skip * 2

# Printing every frame slows the program, so only print occasionally.
debug_print_every = 15
frame_counter = 0

# Resize for performance
frame1 = cv2.resize(frame1, PROCESS_SIZE)
frame2 = cv2.resize(frame2, PROCESS_SIZE)

paused = False

# ================================
# Grid settings
# ================================
# Section 10 uses the optimised skip value above.

# ================================
# Section 4: Tune Lucas-Kanade Settings (Long)
# Press 1, 2, or 3 while program runs
# 1 = stable, 2 = balanced, 3 = sensitive
# ================================
lk_presets = {
    "stable": dict(
        winSize=(25, 25),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            20,
            0.01
        )
    ),

    "balanced": dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03
        )
    ),

    "sensitive": dict(
        winSize=(9, 9),
        maxLevel=1,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            8,
            0.05
        )
    )
}

current_lk_mode = "balanced"
lk_params = lk_presets[current_lk_mode]

# ================================
# Farneback arrow visualisation
# ================================
def draw_farneback_arrows(flow, frame, step):
    output = frame.copy()
    h, w = flow.shape[:2]

    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]

            x2 = int(x + fx)
            y2 = int(y + fy)

            cv2.arrowedLine(
                output,
                (x, y),
                (x2, y2),
                (0, 0, 255),
                1,
                tipLength=0.3
            )

    return output

# Create windows
cv2.namedWindow("Vector Field (Lucas-Kanade)", cv2.WINDOW_NORMAL)
cv2.namedWindow("Divergence Heatmap", cv2.WINDOW_NORMAL)
cv2.namedWindow("Vector Field (Farneback)", cv2.WINDOW_NORMAL)

# Same window sizes
cv2.resizeWindow("Vector Field (Lucas-Kanade)", 640, 360)
cv2.resizeWindow("Divergence Heatmap", 640, 360)
cv2.resizeWindow("Vector Field (Farneback)", 640, 360)

# Put windows beside each other
cv2.moveWindow("Vector Field (Lucas-Kanade)", 50, 100)
cv2.moveWindow("Divergence Heatmap", 750, 100)
cv2.moveWindow("Vector Field (Farneback)", 50, 520)

# ================================
# Section 10: Pre-compute grid points once
# ================================
grid_x = np.arange(0, PROCESS_WIDTH, skip)
grid_y = np.arange(0, PROCESS_HEIGHT, skip)
grid_w = len(grid_x)
grid_h = len(grid_y)
grid_points = np.array(
    [[x, y] for y in grid_y for x in grid_x],
    dtype=np.float32
).reshape(-1, 1, 2)

# ================================
# Section 2 + Section 6 Settings
# ================================
previous_divergence = None
temporal_alpha = 0.75
sobel_pre_blur_kernel = (3, 3)

# ================================
# Section 3: Optical Flow Stability (Norin)
# ================================
previous_u = None
previous_v = None
vector_temporal_alpha = 0.6

# ================================
# Section 7: Max-Div Tracking Stability (Huynh)
# ================================
previous_max_loc = None
max_jump_distance = 80
max_loc_alpha = 0.7

# ================================
# Main loop
# ================================
while True:
    if not paused:
        # ================================
        # Step 1: Convert frames to grayscale
        # ================================
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # ================================
        # Farneback optical flow
        # Section 10: run dense Farneback on smaller frames for speed
        # ================================
        if farneback_scale < 1.0:
            small_gray1 = cv2.resize(
                gray1,
                None,
                fx=farneback_scale,
                fy=farneback_scale,
                interpolation=cv2.INTER_AREA
            )
            small_gray2 = cv2.resize(
                gray2,
                None,
                fx=farneback_scale,
                fy=farneback_scale,
                interpolation=cv2.INTER_AREA
            )

            flow_small = cv2.calcOpticalFlowFarneback(
                small_gray1,
                small_gray2,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.1,
                0
            )

            flow_farneback = cv2.resize(
                flow_small,
                (PROCESS_WIDTH, PROCESS_HEIGHT),
                interpolation=cv2.INTER_LINEAR
            )
            flow_farneback = flow_farneback / farneback_scale
        else:
            flow_farneback = cv2.calcOpticalFlowFarneback(
                gray1,
                gray2,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.1,
                0
            )

        u_fb = flow_farneback[:, :, 0]
        v_fb = flow_farneback[:, :, 1]

        # ================================
        # Section 9: Compare Lucas-Kanade and Farneback (Long)
        # Farneback dense motion magnitude
        # ================================
        fb_magnitude = np.sqrt(u_fb ** 2 + v_fb ** 2)
        fb_avg_motion = np.mean(fb_magnitude)

        du_dy_fb, du_dx_fb = np.gradient(u_fb)
        dv_dy_fb, dv_dx_fb = np.gradient(v_fb)
        divergence_fb = du_dx_fb + dv_dy_fb

        # ================================
        # Step 2: Create uniform grid
        # Section 10: grid is pre-computed once outside the loop
        # ================================
        h, w = gray1.shape

        # ================================
        # Step 3: Lucas kanade Compute optical flow
        # Section 4 uses the selected lk_params preset here
        # ================================
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            gray1,
            gray2,
            grid_points,
            None,
            **lk_params
        )

        if new_points is None or status is None or error is None:
            frame1 = frame2
            ret, frame2 = cap.read()

            if not ret or frame2 is None:
                break

            frame2 = cv2.resize(frame2, PROCESS_SIZE)
            continue

        status = status.flatten()
        error = error.flatten()

        # ================================
        # Step 4: Form vector field (u, v)
        # Section 10: vectorised calculation instead of looping point-by-point
        # ================================
        movement = new_points[:, 0, :] - grid_points[:, 0, :]
        valid_status = status == 1

        u = np.zeros(len(grid_points), dtype=np.float32)
        v = np.zeros(len(grid_points), dtype=np.float32)
        u[valid_status] = movement[valid_status, 0]
        v[valid_status] = movement[valid_status, 1]

        # ================================
        # Section 3: Temporal smoothing for vectors (Team Addition)
        # ================================
        if previous_u is not None and previous_v is not None:
            if previous_u.shape == u.shape and previous_v.shape == v.shape:
                u = vector_temporal_alpha * previous_u + (1 - vector_temporal_alpha) * u
                v = vector_temporal_alpha * previous_v + (1 - vector_temporal_alpha) * v

        previous_u = u.copy()
        previous_v = v.copy()

        # ================================
        # Section 9: Compare Lucas-Kanade and Farneback (Long)
        # Lucas-Kanade valid vector count and average motion
        # ================================
        valid_lk = (status == 1)

        # Use same filtering idea as visualisation
        valid_lk = valid_lk & (error <= 15)
        valid_lk = valid_lk & ~(
            (np.abs(u) < 0.5) &
            (np.abs(v) < 0.5)
        )
        valid_lk = valid_lk & (np.abs(u) <= 30)
        valid_lk = valid_lk & (np.abs(v) <= 30)

        lk_valid_count = np.sum(valid_lk)

        if lk_valid_count > 0:
            lk_magnitude = np.sqrt(
                u[valid_lk] ** 2 +
                v[valid_lk] ** 2
            )
            lk_avg_motion = np.mean(lk_magnitude)
        else:
            lk_avg_motion = 0

        # ================================
        # Step 5: Draw arrow visualisation
        # Part 3: Improve optical flow stability thresholds (Norin)
        # ================================
        vector_output = frame1.copy()

        for i in range(len(grid_points)):
            if status[i] == 0:
                continue

            x1 = int(grid_points[i][0][0])
            y1 = int(grid_points[i][0][1])

            dx = u[i]
            dy = v[i]

            # 1. Stricter Error Filter (Norin)
            if error[i] > 15:
                continue

            # 2. Noise Filter (Norin)
            if abs(dx) < 0.5 and abs(dy) < 0.5:
                continue

            # 3. Anomaly Filter (Norin)
            if abs(dx) > 30 or abs(dy) > 30:
                continue

            x2 = int(x1 + dx)
            y2 = int(y1 + dy)

            cv2.arrowedLine(
                vector_output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                1,
                tipLength=0.3
            )

        # ================================
        # Section 4: Show current LK tuning mode on screen (Long)
        # ================================
        cv2.putText(
            vector_output,
            f"LK Mode: {current_lk_mode}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ================================
        # Section 9: Show LK vs Farneback comparison on screen (Long)
        # ================================
        cv2.putText(
            vector_output,
            f"LK vectors: {lk_valid_count}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            vector_output,
            f"LK Avg Motion: {lk_avg_motion:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            vector_output,
            f"FB Avg Motion: {fb_avg_motion:.2f}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # ================================
        # Farneback arrow visualisation
        # ================================
        farneback_output = draw_farneback_arrows(
            flow_farneback,
            frame1,
            farneback_draw_step
        )

        # ================================
        # Step 6: Compute derivatives using Sobel filter
        # Section 2: Add Sobel filter (Mekno)
        # ================================
        # Section 10: grid_w and grid_h are pre-computed outside the loop.
        u_grid = u.reshape(grid_h, grid_w).astype(np.float32)
        v_grid = v.reshape(grid_h, grid_w).astype(np.float32)

        u_grid_smooth = cv2.GaussianBlur(u_grid, sobel_pre_blur_kernel, 0)
        v_grid_smooth = cv2.GaussianBlur(v_grid, sobel_pre_blur_kernel, 0)

        du_dx = cv2.Sobel(u_grid_smooth, cv2.CV_64F, 1, 0, ksize=3) / skip
        dv_dy = cv2.Sobel(v_grid_smooth, cv2.CV_64F, 0, 1, ksize=3) / skip

        # ================================
        # Step 7: Compute divergence
        # Section 2: Combine Sobel derivatives (Mekno)
        # ================================
        raw_divergence = du_dx + dv_dy

        # ================================
        # Step 7.1: Temporal smoothing
        # Section 6: Add temporal smoothing
        # ================================
        if previous_divergence is None:
            smoothed_divergence = raw_divergence.copy()
        elif previous_divergence.shape != raw_divergence.shape:
            smoothed_divergence = raw_divergence.copy()
        else:
            smoothed_divergence = (
                temporal_alpha * previous_divergence
                + (1 - temporal_alpha) * raw_divergence
            )

        previous_divergence = smoothed_divergence.copy()
        divergence = smoothed_divergence

        # ================================
        # Keep only positive divergence
        # ================================
        positive_divergence = np.maximum(divergence, 0)
        positive_divergence = cv2.GaussianBlur(
            positive_divergence,
            (3, 3),
            0
        )

        # ================================
        # Step 8: Create heatmap visualisation
        # Part 8: Reverted to Original Color Jet Visualization (Norin)
        # ================================
        heatmap_large = cv2.resize(
            positive_divergence,
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )

        heatmap_norm = cv2.normalize(
            heatmap_large,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        heatmap_uint8 = np.uint8(heatmap_norm)

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        heatmap_output = cv2.addWeighted(
            frame1,
            0.6,
            heatmap_color,
            0.4,
            0
        )

        # ================================
        # Step 9: Mark strongest divergence point
        # Section 7: Stabilized max tracking (Huynh)
        # Part 8: Reverted to Original Tracker Visuals (Norin)
        # ================================
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(
            heatmap_uint8
        )

        if max_val > 0:
            if previous_max_loc is None:
                stable_max_loc = max_loc
            else:
                dx = max_loc[0] - previous_max_loc[0]
                dy = max_loc[1] - previous_max_loc[1]
                distance = np.sqrt(dx**2 + dy**2)

                if distance > max_jump_distance:
                    search_radius = max_jump_distance
                    x1 = max(0, previous_max_loc[0] - search_radius)
                    x2 = min(w, previous_max_loc[0] + search_radius)
                    y1 = max(0, previous_max_loc[1] - search_radius)
                    y2 = min(h, previous_max_loc[1] + search_radius)

                    local_region = heatmap_uint8[int(y1):int(y2), int(x1):int(x2)]

                    if local_region.size > 0:
                        _, local_max_val, _, local_max_loc = cv2.minMaxLoc(local_region)
                        local_max_loc = (
                            int(x1 + local_max_loc[0]),
                            int(y1 + local_max_loc[1])
                        )

                        if local_max_val > max_val * 0.5:
                            stable_max_loc = local_max_loc
                        else:
                            stable_max_loc = previous_max_loc
                    else:
                        stable_max_loc = previous_max_loc
                else:
                    stable_max_loc = (
                        int(max_loc_alpha * previous_max_loc[0] + (1 - max_loc_alpha) * max_loc[0]),
                        int(max_loc_alpha * previous_max_loc[1] + (1 - max_loc_alpha) * max_loc[1])
                    )

            previous_max_loc = stable_max_loc

            cv2.circle(
                heatmap_output,
                stable_max_loc,
                10,
                (255, 255, 255),
                2
            )

            cv2.putText(
                heatmap_output,
                "Strongest area",
                (stable_max_loc[0] + 10, stable_max_loc[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        # ================================
        # Debug output
        # ================================
        # Section 10: reduce console printing overhead.
        if frame_counter % debug_print_every == 0:
            print("Section 4 + Section 9 + Section 10 Output")
            print("Current LK Mode:", current_lk_mode)
            print("LK winSize:", lk_params["winSize"])
            print("LK maxLevel:", lk_params["maxLevel"])
            print("Lucas-Kanade valid vectors:", lk_valid_count)
            print("Lucas-Kanade average motion:", round(lk_avg_motion, 4))
            print("Farneback average motion:", round(fb_avg_motion, 4))
            print("Optimised frame size:", PROCESS_SIZE)
            print("Optimised grid skip:", skip)
            print("Farneback scale:", farneback_scale)
            print("-" * 40)

        # ================================
        # Display windows
        # ================================
        cv2.imshow("Vector Field (Lucas-Kanade)", vector_output)
        cv2.imshow("Divergence Heatmap", heatmap_output)
        cv2.imshow("Vector Field (Farneback)", farneback_output)

        frame_counter += 1

        # ================================
        # Move to next frame
        # ================================
        frame1 = frame2
        ret, frame2 = cap.read()

        if not ret or frame2 is None:
            break

        frame2 = cv2.resize(frame2, PROCESS_SIZE)

    # ================================
    # Controls
    # ================================
    key = cv2.waitKey(30) & 0xFF

    if key == 27:  # ESC to exit
        break
    elif key == 32:  # SPACE to pause
        paused = not paused

    # ================================
    # Section 4: LK Tuning Controls (Long)
    # ================================
    elif key == ord("1"):
        current_lk_mode = "stable"
        lk_params = lk_presets[current_lk_mode]
        print("Lucas-Kanade Mode = STABLE")

    elif key == ord("2"):
        current_lk_mode = "balanced"
        lk_params = lk_presets[current_lk_mode]
        print("Lucas-Kanade Mode = BALANCED")

    elif key == ord("3"):
        current_lk_mode = "sensitive"
        lk_params = lk_presets[current_lk_mode]
        print("Lucas-Kanade Mode = SENSITIVE")

# ================================
# Cleanup
# ================================
cap.release()
cv2.destroyAllWindows()