import cv2
import numpy as np

cap = cv2.VideoCapture("Foward_Fast(Textured).mp4")

ret, frame1 = cap.read()
ret, frame2 = cap.read()

if not ret or frame1 is None or frame2 is None:
    print("Error: Could not read video.")
    cap.release()
    exit()

frame1 = cv2.resize(frame1, (640, 360))
frame2 = cv2.resize(frame2, (640, 360))

paused = False

#Grid skip
skip = 20
# Lucas–Kanade parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

while True:
    if not paused:
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Create grid
        h, w = gray1.shape
        grid_points = []

        for y in range(0, h, skip):
            for x in range(0, w, skip):
                grid_points.append([x, y])

        grid_points = np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2)

        # Compute Lucas-Kanade optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            gray1,
            gray2,
            grid_points,
            None,
            **lk_params
        )

        if new_points is not None and status is not None:
            good_old = grid_points[status.flatten() == 1]
            good_new = new_points[status.flatten() == 1]
            # u = horizontal motion
            # v = vertical motion
            u = good_new[:, 0, 0] - good_old[:, 0, 0]
            v = good_new[:, 0, 1] - good_old[:, 0, 1]
            points = good_old.reshape(-1, 2)

            flow_field = np.zeros((h, w, 2), dtype=np.float32)

            for (x, y), du, dv in zip(points.astype(int), u, v):
                flow_field[y, x, 0] = du
                flow_field[y, x, 1] = dv

            print("Flow Field")
            print("points:", points.shape)
            print("u:", u.shape)
            print("v:", v.shape)
            print("flow_field:", flow_field.shape)
            print("-" * 40)
        #Show original frame
        cv2.imshow("Frame", frame1)

        # Shift frames
        frame1 = frame2
        ret, frame2 = cap.read()

        if not ret or frame2 is None:
            break

        frame2 = cv2.resize(frame2, (640, 360))

    key = cv2.waitKey(100) & 0xFF

    if key == 27:
        break
    elif key == 32:
        paused = not paused

cap.release()
cv2.destroyAllWindows()