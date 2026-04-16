import cv2
import numpy as np

# ==============================================================================
# SECTION 1: CONFIGURATION & PARAMETERS
# ==============================================================================
video_path = "demo_grass.mp4"
cap = cv2.VideoCapture(video_path)

### -------------------------------------------------------------------------
### TALKING POINT 1: SHI-TOMASI CORNER DETECTION PARAMETERS
### -------------------------------------------------------------------------
feature_params = dict(
    maxCorners=100,      # Maximum number of points to track
    qualityLevel=0.3,    # Minimum quality of the corners
    minDistance=7,       # Minimum distance between corners
    blockSize=7
)

### -------------------------------------------------------------------------
### TALKING POINT 2: LUCAS-KANADE PARAMETERS
### -------------------------------------------------------------------------
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)


# ==============================================================================
# SECTION 2: INITIALIZATION (FIRST FRAME)
# ==============================================================================
ret, old_frame = cap.read()
if not ret:
    print("Error: Video not found or failed to load.")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_gray = cv2.GaussianBlur(old_gray, (5, 5), 0)

### -------------------------------------------------------------------------
### TALKING POINT 3: EXTRACTING INITIAL FEATURES
### -------------------------------------------------------------------------
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

mask = np.zeros_like(old_frame)
paused = False
last_valid_img = old_frame.copy() # Keeps the screen visible when auto-frozen

print("CONTROLS: [Space] Pause/Play | [r] Reset Points | [c] Clear Mask | [q] Quit")


# ==============================================================================
# SECTION 3: MAIN TRACKING LOOP
# ==============================================================================
while True:
    if not paused:
        ret, frame = cap.read()

        # Auto-freeze on the final frame for the demo presentation
        if not ret:
            paused = True
            print("Demo Frozen: Ready to explain Max-Divergence or Feature Drift.")
            continue

        ### -----------------------------------------------------------------
        ### TALKING POINT 4: NOISE REDUCTION
        ### (Explain why blurring is critical for real-world robotics)
        ### -----------------------------------------------------------------
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

        ### -----------------------------------------------------------------
        ### TALKING POINT 5: LUCAS-KANADE OPTICAL FLOW CALCULATION
        ### -----------------------------------------------------------------
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)


        # ======================================================================
        # SECTION 4: VISUALIZATION (DRAWING VECTORS)
        # ======================================================================
        if p1 is not None and p0 is not None:
            # Keep only the points that the algorithm successfully tracked
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Draw the tracks (green lines for history, red dots for current position)
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)

            # Overlay the vector lines onto the original video frame
            last_valid_img = cv2.add(frame, mask)

            # Update previous frame and points for the next loop iteration
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    # Display the result (uses last_valid_img so the freeze frame stays on screen)
    cv2.imshow("Lucas-Kanade Vectors", last_valid_img)


    # ==============================================================================
    # SECTION 5: KEYBOARD CONTROLS & CLEANUP
    # ==============================================================================
    key = cv2.waitKey(30) & 0xff

    if key == ord("q"):  # Quit
        break
    elif key == ord(" "):  # Spacebar to Pause/Play
        paused = not paused
    elif key == ord("c"):  # Clear the green vector history lines
        mask = np.zeros_like(old_frame)
    elif key == ord("r"):  # Reset tracking points
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

cap.release()
cv2.destroyAllWindows()