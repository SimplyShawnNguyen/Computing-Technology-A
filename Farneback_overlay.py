import cv2
import numpy as np

# ================================
# Ask user: camera or video?
# ================================
choice = input("Use camera or video? (c/v): ").lower()

if choice == "c":
    # Use webcam (0 = default camera)
    cap = cv2.VideoCapture(0)
elif choice == "v":
    # Ask for video file name
    video_path = input("Enter video file name (e.g. test.mp4): ")
    cap = cv2.VideoCapture(video_path)
else:
    print("Invalid choice. Please enter 'c' or 'v'.")
    exit()

# ================================
# Read first frame (starting point)
# ================================
ret, frame1 = cap.read()

# If it fails, stop the program
if not ret:
    print("Failed to get video/camera input")
    exit()

# Convert first frame to grayscale (easier for processing)
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# ================================
# Main loop (runs every frame)
# ================================
while True:
    ret, frame2 = cap.read()

    # If video ends or camera fails, stop
    if not ret:
        print("End of stream or camera error")
        break

    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # ================================
    # Calculate dense optical flow (Farneback)
    # This gives motion for EVERY pixel
    # ================================
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Split flow into magnitude (speed) and angle (direction)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # ================================
    # Convert flow to color (for visualization)
    # ================================

    # Create HSV image
    hsv = np.zeros_like(frame2)

    # Convert angle to degrees
    angle_deg = ang * 180 / np.pi

    # Set saturation (full color)
    hsv[..., 1] = 255

    # Initialize hue
    hsv[..., 0] = 0

    # Map direction → hue ONLY
    hsv[..., 0][(angle_deg >= 0) & (angle_deg < 90)] = 0      # red
    hsv[..., 0][(angle_deg >= 90) & (angle_deg < 180)] = 60   # green
    hsv[..., 0][(angle_deg >= 180) & (angle_deg < 270)] = 120 # blue
    hsv[..., 0][(angle_deg >= 270)] = 30                      # yellow

    # Set brightness (speed)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV → BGR (needed for display)
    flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # ================================
    # Overlay flow on original frame
    # ================================
    overlay = cv2.addWeighted(frame2, 0.7, flow_color, 0.3, 0)

    # Show result
    cv2.imshow("Farneback Overlay", overlay)

    # ================================
    # Controls
    # ================================
    key = cv2.waitKey(30) & 0xFF

    if key == 27:  # ESC key
        break

    # Update previous frame for next loop
    prev_gray = gray

# ================================
# Cleanup (close everything)
# ================================
cap.release()
cv2.destroyAllWindows()