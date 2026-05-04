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
skip = 15  # Tighter grid for smoother heatmaps

# ================================
# Lucas-Kanade parameters (Upgraded for Fast Movement)
# ================================
lk_params = dict(
    winSize=(31, 31),  # Larger window to catch fast jumps
    maxLevel=4,        # More zoom-out levels for macro-movement
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
        # Step 4: Form vector field (u, v) grids
        # ================================
        u = np.zeros(len(grid_points))
        v = np.zeros(len(grid_points))

        # Filter out bad points based on status and error
        for i in range(len(grid_points)):
            if status[i] == 1 and error[i] <= 20:
                dx = new_points[i, 0, 0] - grid_points[i, 0, 0]
                dy = new_points[i, 0, 1] - grid_points[i, 0, 1]
                
                # Filter out huge anomalies or tiny movements
                if (abs(dx) > 1 or abs(dy) > 1) and (abs(dx) <= 15 and abs(dy) <= 15):
                    u[i] = dx
                    v[i] = dy

        # Reshape flat arrays into 2D grids representing the screen
        grid_w = len(range(0, w, skip))
        grid_h = len(range(0, h, skip))

        u_grid = u.reshape(grid_h, grid_w)
        v_grid = v.reshape(grid_h, grid_w)

        # ================================
        # Step 4.5: Generate Heatmap Visualization
        # ================================
        # 1. Calculate magnitude (speed) of vectors
        magnitude = np.sqrt(u_grid**2 + v_grid**2)
        
        # 2. Smooth the tiny grid BEFORE resizing to blend the sparse data points
        magnitude_smoothed = cv2.GaussianBlur(magnitude, (3, 3), 0)
        
        # 3. Normalize to 0-255 for coloring
        mag_normalized = cv2.normalize(magnitude_smoothed, None, 0, 255, cv2.NORM_MINMAX)
        mag_uint8 = np.uint8(mag_normalized)
        
        # 4. Resize up to the full video frame size (640x360) smoothly
        heatmap_intensity = cv2.resize(mag_uint8, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # 5. Apply a colormap
        heatmap_color = cv2.applyColorMap(heatmap_intensity, cv2.COLORMAP_JET)

        # 6. DYNAMIC TRANSPARENCY
        # Convert intensity to a 0.0 to 1.0 scale for transparency
        alpha = heatmap_intensity.astype(float) / 255.0  
        
        # Apply a power curve to crush background noise and boost the main signals
        alpha = np.power(alpha, 2.5) 
        
        # Convert alpha to 3 channels so it can blend with the BGR image
        alpha_3d = cv2.merge([alpha, alpha, alpha])
        
        # 7. Blend the images
        max_opacity = 0.85 # 85% maximum solidness for the colors
        output_float = frame1.astype(float) * (1.0 - (alpha_3d * max_opacity)) + heatmap_color.astype(float) * (alpha_3d * max_opacity)
        output = np.uint8(output_float)

        # ================================
        # Step 5: Compute derivatives du/dx and dv/dy
        # ================================
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
        cv2.imshow("Heatmap (Lucas-Kanade)", output)

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