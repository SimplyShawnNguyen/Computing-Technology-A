import cv2
import numpy as np

# ================================
# Load video
# ================================
cap = cv2.VideoCapture("Foward_Fast(Textured).mp4")

ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Resize for performance
frame1 = cv2.resize(frame1, (640, 360))
frame2 = cv2.resize(frame2, (640, 360))

paused = False

# ================================
# Grid settings (Step 2)
# ================================
skip = 35  # distance between grid points

# ================================
# Lucas-Kanade parameters (Step 3)
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
        # Step 2: Create uniform grid
        # ================================
        h, w = gray1.shape
        grid_points = []

        for y in range(0, h, skip):
            for x in range(0, w, skip):
                grid_points.append([x, y])

        grid_points = np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2)

        # ================================
        # Step 3: Compute optical flow (Lucas–Kanade)
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

        # Draw arrows on frame (vector field visualisation)
        output = frame1.copy()

        for i in range(len(grid_points)):
            if status[i] == 0:
                continue

            x1, y1 = int(grid_points[i][0][0]), int(grid_points[i][0][1])

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

            cv2.arrowedLine(output, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)

        # ================================
        # Step 5: Compute derivatives du/dx and dv/dy
        # ================================
        grid_w = len(range(0, w, skip))
        grid_h = len(range(0, h, skip))

        u_grid = u.reshape(grid_h, grid_w)
        v_grid = v.reshape(grid_h, grid_w)

        du_dy, du_dx = np.gradient(u_grid, skip, skip)
        dv_dy, dv_dx = np.gradient(v_grid, skip, skip)

        # ================================
        # Step 6: Compute divergence D = du/dx + dv/dy
        # ================================
        divergence = du_dx + dv_dy

        # Debug output
        print("Section 6 Output")
        print("divergence shape:", divergence.shape)
        print("divergence min:", round(divergence.min(), 4), "  max:", round(divergence.max(), 4))
        print("-" * 40)

        # ================================
        # Display result
        # ================================
        cv2.imshow("Vector Field (Lucas-Kanade)", output)

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