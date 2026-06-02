import cv2
import numpy as np

# ================================
# Load video
# ================================
cap = cv2.VideoCapture("BlackWall.mp4")

ret, frame1 = cap.read()
ret, frame2 = cap.read()

if not ret or frame1 is None or frame2 is None:
    print("Error: Could not read video.")
    exit()

# ================================
# Section 10: Performance Optimisation (Van Tuy)
# ================================
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360
PROCESS_SIZE = (PROCESS_WIDTH, PROCESS_HEIGHT)

skip = 40

farneback_scale = 0.5
farneback_draw_step = skip * 2

debug_print_every = 15
frame_counter = 0

frame1 = cv2.resize(frame1, PROCESS_SIZE)
frame2 = cv2.resize(frame2, PROCESS_SIZE)

paused = False

# ================================
# Vector filter settings
# ================================
min_vector_magnitude = 0.5
max_vector_magnitude = 30.0
max_vector_component = 30.0
max_lk_error = 15.0

def get_vector_mask(u, v):
    magnitude = np.sqrt(u ** 2 + v ** 2)

    valid_mask = np.isfinite(u) & np.isfinite(v)
    valid_mask = valid_mask & (magnitude >= min_vector_magnitude)
    valid_mask = valid_mask & (magnitude <= max_vector_magnitude)
    valid_mask = valid_mask & (np.abs(u) <= max_vector_component)
    valid_mask = valid_mask & (np.abs(v) <= max_vector_component)

    return valid_mask, magnitude

# ================================
# Section 4: Tune Lucas-Kanade Settings (Long)
# ================================
lk_presets = {
    "stable": dict(
        winSize=(31, 31),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            25,
            0.01
        )
    ),

    "balanced": dict(
        winSize=(21, 21),
        maxLevel=2,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            15,
            0.02
        )
    ),

    "sensitive": dict(
        winSize=(15, 15),
        maxLevel=1,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03
        )
    )
}

current_lk_mode = "stable"
lk_params = lk_presets[current_lk_mode]

# ================================
# Farneback arrow visualisation
# ================================
def draw_farneback_arrows(flow, frame, step, valid_mask):
    output = frame.copy()
    h, w = flow.shape[:2]

    for y in range(0, h, step):
        for x in range(0, w, step):
            if not valid_mask[y, x]:
                continue

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

def get_patch_max_location(heatmap_uint8, patch_size=31):
    patch_average = cv2.blur(heatmap_uint8, (patch_size, patch_size))
    _, max_val, _, max_loc = cv2.minMaxLoc(patch_average)
    return max_val, max_loc

# Create windows
cv2.namedWindow("Vector Field (Lucas-Kanade)", cv2.WINDOW_NORMAL)
cv2.namedWindow("Divergence Heatmap", cv2.WINDOW_NORMAL)
cv2.namedWindow("Vector Field (Farneback)", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Vector Field (Lucas-Kanade)", 640, 360)
cv2.resizeWindow("Divergence Heatmap", 640, 360)
cv2.resizeWindow("Vector Field (Farneback)", 640, 360)

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
max_jump_distance = 120
max_loc_alpha = 0.75

# ================================
# Main loop
# ================================
while True:
    if not paused:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # ================================
        # Farneback optical flow
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
        # Farneback vector filtering
        # ================================
        fb_valid_mask, fb_magnitude = get_vector_mask(u_fb, v_fb)

        u_fb_filtered = np.where(fb_valid_mask, u_fb, 0)
        v_fb_filtered = np.where(fb_valid_mask, v_fb, 0)

        flow_farneback_filtered = np.dstack(
            (u_fb_filtered, v_fb_filtered)
        )

        if np.any(fb_valid_mask):
            fb_avg_motion = np.mean(fb_magnitude[fb_valid_mask])
        else:
            fb_avg_motion = 0

        du_dy_fb, du_dx_fb = np.gradient(u_fb_filtered)
        dv_dy_fb, dv_dx_fb = np.gradient(v_fb_filtered)
        divergence_fb = du_dx_fb + dv_dy_fb

        # ================================
        # Lucas-Kanade optical flow
        # ================================
        h, w = gray1.shape

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

        movement = new_points[:, 0, :] - grid_points[:, 0, :]
        valid_status = status == 1

        u = np.zeros(len(grid_points), dtype=np.float32)
        v = np.zeros(len(grid_points), dtype=np.float32)
        u[valid_status] = movement[valid_status, 0]
        v[valid_status] = movement[valid_status, 1]

        # ================================
        # Section 3: Temporal smoothing for vectors
        # ================================
        if previous_u is not None and previous_v is not None:
            if previous_u.shape == u.shape and previous_v.shape == v.shape:
                u = vector_temporal_alpha * previous_u + (1 - vector_temporal_alpha) * u
                v = vector_temporal_alpha * previous_v + (1 - vector_temporal_alpha) * v

        previous_u = u.copy()
        previous_v = v.copy()

        # ================================
        # Lucas-Kanade vector filtering
        # ================================
        valid_lk = status == 1

        lk_size_mask, lk_magnitude_all = get_vector_mask(u, v)

        valid_lk = valid_lk & (error <= max_lk_error)
        valid_lk = valid_lk & lk_size_mask

        lk_valid_count = np.sum(valid_lk)

        if lk_valid_count > 0:
            lk_avg_motion = np.mean(lk_magnitude_all[valid_lk])
        else:
            lk_avg_motion = 0

        # ================================
        # Draw Lucas-Kanade arrow visualisation
        # ================================
        vector_output = frame1.copy()

        for i in range(len(grid_points)):
            if not valid_lk[i]:
                continue

            x1 = int(grid_points[i][0][0])
            y1 = int(grid_points[i][0][1])

            dx = u[i]
            dy = v[i]

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

        cv2.putText(
            vector_output,
            f"LK Mode: {current_lk_mode}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

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
        # Draw Farneback arrow visualisation
        # ================================
        farneback_output = draw_farneback_arrows(
            flow_farneback_filtered,
            frame1,
            farneback_draw_step,
            fb_valid_mask
        )

        # ================================
        # Compute derivatives using Sobel filter
        # ================================
        u_filtered = np.where(valid_lk, u, 0)
        v_filtered = np.where(valid_lk, v, 0)

        u_grid = u_filtered.reshape(grid_h, grid_w).astype(np.float32)
        v_grid = v_filtered.reshape(grid_h, grid_w).astype(np.float32)

        u_grid_smooth = cv2.GaussianBlur(u_grid, sobel_pre_blur_kernel, 0)
        v_grid_smooth = cv2.GaussianBlur(v_grid, sobel_pre_blur_kernel, 0)

        du_dx = cv2.Sobel(u_grid_smooth, cv2.CV_64F, 1, 0, ksize=7) / skip
        dv_dy = cv2.Sobel(v_grid_smooth, cv2.CV_64F, 0, 1, ksize=7) / skip

        raw_divergence = du_dx + dv_dy

        # ================================
        # Temporal smoothing
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

        positive_divergence = np.maximum(divergence, 0)
        positive_divergence = cv2.GaussianBlur(
            positive_divergence,
            (3, 3),
            0
        )

        # ================================
        # Create heatmap visualisation
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
        # Centre/reference point always drawn
        # ================================
        center_point = (w // 2, h // 2)

        cv2.drawMarker(
            heatmap_output,
            center_point,
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=45,
            thickness=3
        )

        cv2.drawMarker(
            vector_output,
            center_point,
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=45,
            thickness=3
        )

        cv2.drawMarker(
            farneback_output,
            center_point,
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=45,
            thickness=3
        )

        # ================================
        # Patch-based strongest divergence point
        # ================================
        patch_size = 51
        search_radius = 90

        if previous_max_loc is None:
            max_val, max_loc = get_patch_max_location(
                heatmap_uint8,
                patch_size=patch_size
            )
        else:
            px, py = previous_max_loc

            x1 = max(0, px - search_radius)
            x2 = min(w, px + search_radius)
            y1 = max(0, py - search_radius)
            y2 = min(h, py + search_radius)

            local_heatmap = heatmap_uint8[y1:y2, x1:x2]

            if local_heatmap.size > 0:
                max_val, local_max_loc = get_patch_max_location(
                    local_heatmap,
                    patch_size=patch_size
                )

                max_loc = (
                    x1 + local_max_loc[0],
                    y1 + local_max_loc[1]
                )
            else:
                max_val = 0
                max_loc = previous_max_loc

        if max_val > 0:
            if previous_max_loc is None:
                stable_max_loc = max_loc
            else:
                stable_max_loc = (
                    int(0.85 * previous_max_loc[0] + 0.15 * max_loc[0]),
                    int(0.85 * previous_max_loc[1] + 0.15 * max_loc[1])
                )

            previous_max_loc = stable_max_loc

            cv2.drawMarker(
                heatmap_output,
                stable_max_loc,
                (255, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=45,
                thickness=3
            )

            cv2.drawMarker(
                vector_output,
                stable_max_loc,
                (255, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=45,
                thickness=3
            )

            cv2.drawMarker(
                farneback_output,
                stable_max_loc,
                (255, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=45,
                thickness=3
            )

        # ================================
        # Debug output
        # ================================
        if frame_counter % debug_print_every == 0:
            print("Section 4 + Section 9 + Section 10 Output")
            print("Current LK Mode:", current_lk_mode)
            print("LK winSize:", lk_params["winSize"])
            print("LK maxLevel:", lk_params["maxLevel"])
            print("Lucas-Kanade valid vectors:", lk_valid_count)
            print("Lucas-Kanade average motion:", round(lk_avg_motion, 4))
            print("Farneback average motion:", round(fb_avg_motion, 4))
            print("Farneback valid pixels:", np.sum(fb_valid_mask))
            print("Optimised frame size:", PROCESS_SIZE)
            print("Optimised grid skip:", skip)
            print("Farneback scale:", farneback_scale)

            if previous_max_loc is not None:
                print("MaxDiv location:", previous_max_loc)
                print("MaxDiv offset from centre:", (
                    previous_max_loc[0] - center_point[0],
                    previous_max_loc[1] - center_point[1]
                ))

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

    if key == 27:
        break
    elif key == 32:
        paused = not paused

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