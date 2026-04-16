import cv2
import numpy as np

# Camera Setup
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open Mac camera.")
    exit()

# Feature Detection Parameters
feature_params = dict(
    maxCorners=40,
    qualityLevel=0.3,
    minDistance=10,
    blockSize=7
)

# Lucas-Kanade Optical Flow Parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Read First Frame
ret, old_frame = cap.read()
if not ret:
    print("Error: Could not read first frame.")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask = np.zeros_like(old_frame)

# State for display
frame_count = 0
display_divergence = 0.0
display_text = "Stable"

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_count += 1

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2

    # Fade old motion lines gradually
    mask = (mask * 0.9).astype(np.uint8)

    # Re-detect points if lost
    if p0 is None or len(p0) == 0:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    divergence_value = 0.0

    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            radial_changes = []
            filtered_new = []

            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()

                # Focus more on the center area
                if abs(c - cx) > w * 0.35:
                    continue

                # Ignore tiny noisy motion
                movement = np.sqrt((a - c) ** 2 + (b - d) ** 2)
                if movement < 2:
                    continue

                a_i, b_i = int(a), int(b)
                c_i, d_i = int(c), int(d)

                # Draw motion
                cv2.line(mask, (c_i, d_i), (a_i, b_i), (0, 255, 0), 1)
                cv2.circle(frame, (a_i, b_i), 3, (0, 0, 255), -1)

                # Radial change from image center
                old_r = np.sqrt((c - cx) ** 2 + (d - cy) ** 2)
                new_r = np.sqrt((a - cx) ** 2 + (b - cy) ** 2)
                radial_changes.append(new_r - old_r)

                filtered_new.append([[a, b]])

            if len(radial_changes) > 0:
                divergence_value = float(np.mean(radial_changes))

                # Update displayed result only every 5 frames
            if frame_count % 10 == 0:
                    display_divergence = divergence_value

                    if divergence_value > 0.5:
                        display_text = "Approaching surface"
                    elif divergence_value < -0.5:
                        display_text = "Moving away"
                    else:
                        display_text = "Stable"
            else:
                # If no strong motion, keep last displayed value/text
                pass

            output = cv2.add(frame, mask)

            # Stronger center marker
            cv2.circle(output, (int(cx), int(cy)), 6, (255, 0, 0), -1)
            cv2.circle(output, (int(cx), int(cy)), 20, (255, 0, 0), 1)

            # Info panel background
            cv2.rectangle(output, (10, 10), (360, 100), (0, 0, 0), -1)

            # Display text clearly
            cv2.putText(
                output,
                f"Divergence: {display_divergence:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                output,
                display_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.imshow("Lucas-Kanade with Divergence", output)

            old_gray = frame_gray.copy()

            if len(filtered_new) > 0:
                p0 = np.array(filtered_new, dtype=np.float32)
            else:
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

        else:
            cv2.imshow("Lucas-Kanade with Divergence", frame)
            old_gray = frame_gray.copy()
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            mask = np.zeros_like(frame)
    else:
        cv2.imshow("Lucas-Kanade with Divergence", frame)
        old_gray = frame_gray.copy()

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        mask = np.zeros_like(frame)
        old_gray = frame_gray.copy()
        frame_count = 0
        display_divergence = 0.0
        display_text = "Stable"

cap.release()
cv2.destroyAllWindows()