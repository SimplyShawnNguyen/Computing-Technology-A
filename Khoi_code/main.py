import cv2
import numpy as np

cap = cv2.VideoCapture("Foward_Fast(Textured).mp4")

ret, frame1 = cap.read()
ret, frame2 = cap.read()

frame1 = cv2.resize(frame1, (640, 360))
frame2 = cv2.resize(frame2, (640, 360))

paused = False

while True:
    if not paused:
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            0.5, 3, 15, 3, 5, 1.1, 0
        )

        # Draw arrows for visualization
        output = frame1.copy()
        step = 30
        h, w = gray1.shape

        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]
                end_x = int(x + fx)
                end_y = int(y + fy)
                cv2.arrowedLine(output, (x, y), (end_x, end_y), (0, 255, 0), 1, tipLength=0.3)

        # Show windows
        cv2.imshow("Original", frame1)
        cv2.imshow("Optical Flow", output)

        # Shift frames
        frame1 = frame2
        ret, frame2 = cap.read()

        if not ret or frame2 is None:
            break

        frame2 = cv2.resize(frame2, (640, 360))

    key = cv2.waitKey(30) & 0xFF

    if key == 27:
        break
    elif key == 32:  #Paused  video for better view
        paused = not paused  

cap.release()
cv2.destroyAllWindows()