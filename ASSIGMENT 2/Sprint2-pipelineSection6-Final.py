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

        # ================================
        # Farneback: Form vector field (u, v)
        # ================================
        u_fb = flow_farneback[:, :, 0]
        v_fb = flow_farneback[:, :, 1]

        # ================================
        # Farneback: Compute derivatives
        # ================================
        du_dy_fb, du_dx_fb = np.gradient(u_fb)
        dv_dy_fb, dv_dx_fb = np.gradient(v_fb)

        # ================================
        # Farneback: Compute divergence
        # ================================
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
        # Step 5: Draw arrow visualisation
        # ================================
        vector_output = frame1.copy()

        for i in range(len(grid_points)):
            if status[i] == 0:
                continue

            x1 = int(grid_points[i][0][0])
            y1 = int(grid_points[i][0][1])

            dx = u[i]
            dy = v[i]

            if error[i] > 20:
                continue

            if abs(dx) < 1 and abs(dy) < 1:
                continue

            if abs(dx) > 15 or abs(dy) > 15:
                continue

            scale = 1

            x2 = int(x1 + dx * scale)
            y2 = int(y1 + dy * scale)

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
        # Step 6: Compute derivatives
        # ================================
        grid_w = len(range(0, w, skip))
        grid_h = len(range(0, h, skip))

        u_grid = u.reshape(grid_h, grid_w)
        v_grid = v.reshape(grid_h, grid_w)

        du_dy, du_dx = np.gradient(u_grid, skip, skip)
        dv_dy, dv_dx = np.gradient(v_grid, skip, skip)

        # ================================
        # Step 7: Compute divergence
        # ================================
        divergence = du_dx + dv_dy

        # Keep only positive divergence
        positive_divergence = np.maximum(divergence, 0)

        # Smooth it slightly
        positive_divergence = cv2.GaussianBlur(
            positive_divergence,
            (3, 3),
            0
        )

        # ================================
        # Step 8: Create heatmap visualisation
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
        # ================================
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(
            heatmap_uint8
        )

        if max_val > 0:
            cv2.circle(
                heatmap_output,
                max_loc,
                10,
                (255, 255, 255),
                2
            )

            cv2.putText(
                heatmap_output,
                "Strongest area",
                (max_loc[0] + 10, max_loc[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        # ================================
        # Debug output
        # ================================
        print("Section 6 Output")
        print("Lucas-Kanade divergence shape:", divergence.shape)
        print(
            "Lucas-Kanade divergence min:",
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