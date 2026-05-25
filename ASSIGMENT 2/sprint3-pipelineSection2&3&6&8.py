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

# Resize for performance
frame1 = cv2.resize(frame1, (640, 360))
frame2 = cv2.resize(frame2, (640, 360))

paused = False

# ================================
# Grid settings
# ================================
skip = 35  # distance between grid points

# ================================
# Lucas-Kanade parameters
# ================================
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

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
        # ================================
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
        du_dy_fb, du_dx_fb = np.gradient(u_fb)
        dv_dy_fb, dv_dx_fb = np.gradient(v_fb)
        divergence_fb = du_dx_fb + dv_dy_fb

        # ================================
        # Step 2: Create uniform grid
        # ================================
        h, w = gray1.shape
        grid_points = []

        for y in range(0, h, skip):
            for x in range(0, w, skip):
                grid_points.append([x, y])

        grid_points = np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2)

        # ================================
        # Step 3: Lucas kanade Compute optical flow
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

            frame2 = cv2.resize(frame2, (640, 360))
            continue

        status = status.flatten()
        error = error.flatten()

        # ================================
        # Step 4: Form vector field (u, v)
        # ================================
        u = np.zeros(len(grid_points))
        v = np.zeros(len(grid_points))

        for i in range(len(grid_points)):
            if status[i] == 1:
                u[i] = new_points[i, 0, 0] - grid_points[i, 0, 0]
                v[i] = new_points[i, 0, 1] - grid_points[i, 0, 1]

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
        # Farneback arrow visualisation
        # ================================
        farneback_output = draw_farneback_arrows(
            flow_farneback,
            frame1,
            skip
        )

        # ================================
        # Step 6: Compute derivatives using Sobel filter
        # Section 2: Add Sobel filter (Mekno)
        # ================================
        grid_w = len(range(0, w, skip))
        grid_h = len(range(0, h, skip))

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
            
            # --- Part 8: Draw the original circle and text ---
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
        print("Section 2 + Section 6 Output")
        print("Sobel raw divergence shape:", raw_divergence.shape)
        print(
            "Sobel raw divergence min:",
            round(raw_divergence.min(), 4),
            " max:",
            round(raw_divergence.max(), 4)
        )

        print(
            "Temporal smoothed divergence min:",
            round(divergence.min(), 4),
            " max:",
            round(divergence.max(), 4)
        )

        print("Farneback divergence shape:", divergence_fb.shape)
        print(
            "Farneback divergence min:",
            round(divergence_fb.min(), 4),
            " max:",
            round(divergence_fb.max(), 4)
        )

        print("-" * 40)

        # ================================
        # Display windows
        # ================================
        cv2.imshow("Vector Field (Lucas-Kanade)", vector_output)
        cv2.imshow("Divergence Heatmap", heatmap_output)
        cv2.imshow("Vector Field (Farneback)", farneback_output)

        # ================================
        # Move to next frame
        # ================================
        frame1 = frame2
        ret, frame2 = cap.read()

        if not ret or frame2 is None:
            break

        frame2 = cv2.resize(frame2, (640, 360))

    # ================================
    # Controls
    # ================================
    key = cv2.waitKey(30) & 0xFF

    if key == 27:  # ESC to exit
        break
    elif key == 32:  # SPACE to pause
        paused = not paused

# ================================
# Cleanup
# ================================
cap.release()
cv2.destroyAllWindows()