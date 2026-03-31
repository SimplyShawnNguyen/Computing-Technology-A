📌 Vision-Only Surface Alignment Using Optical Flow Divergence (WILL FIX ALONG THE PROJECT PROGRESS)
📖 Project Overview

This project focuses on developing a vision-only system for surface alignment and docking using optical flow. The system estimates motion directly from camera input without relying on additional sensors such as LiDAR or depth cameras.

The project is based on the concept of optical flow divergence, which provides information about how close the camera is to a surface (time-to-contact). More advanced approaches, such as maximum divergence (max-div), allow control of both approach speed and direction for docking tasks .

🎯 Project Goals
Estimate motion using camera input only
Detect approach toward surfaces using optical flow
Understand expansion (forward motion) and contraction (backward motion)
Analyse behaviour under different conditions (texture, motion type)
Build a foundation for robot docking and surface alignment
🧠 Key Concepts
🔹 Optical Flow

Tracks how pixels move between frames to estimate motion.

🔹 Divergence

Measures how much the flow expands or contracts:

Expansion → moving toward surface
Contraction → moving away
🔹 Max-Divergence (Max-Div)

A key concept from the research paper:

The point of maximum divergence indicates important motion information
Can be used to control approach speed and direction
Enables docking without knowing exact surface orientation
🛠️ Technologies Used
Python
OpenCV
NumPy
⚙️ System Features
Real-time webcam input
Frame preprocessing (resize + grayscale)
Optical flow computation:
Farneback (dense)
Lucas–Kanade (sparse)
Motion visualisation:
Vector arrows
Overlay
Heatmaps
🧪 How the System Works
Capture two consecutive frames
Convert frames to grayscale
Compute optical flow between frames
Extract motion vectors (fx, fy)
Visualise motion using arrows or other methods
Repeat continuously for real-time output
📊 Key Findings (Sprint 1)
Motion detection works well in real time
Object movement is easier to detect than camera forward/backward motion
Optical flow direction is opposite to camera movement
Flat (low-texture) surfaces produce weak and noisy results
Faster movement improves motion visibility slightly
⚠️ Limitations
Poor performance on flat or featureless surfaces
Noise in low-texture regions
Forward/backward motion is harder to detect
Sensitive to lighting and small motion
🔍 Future Work
Implement divergence calculation (div = ux + vy)
Detect max-div point in the flow field
Use divergence for time-to-contact estimation
Apply control laws for robot docking
Integrate with real robotic system (CP system / PLC + BaSyx)
👥 Team Approach

Each team member implemented a different combination of:

Optical flow method (Farneback / Lucas–Kanade)
Visualisation type (vectors / overlay / heatmap)

Results were compared to analyse performance and behaviour.

📌 Project Significance

This project demonstrates that:

Vision-only systems can estimate motion and proximity
Optical flow can support robot control tasks
However, performance depends heavily on environmental conditions

It also provides a foundation for advanced approaches like max-div, which enables more robust docking and landing strategies without requiring full 3D reconstruction .

📸 Example Output

<img width="1219" height="731" alt="{B5F8C847-954E-4FE3-8F9B-F459BA336DE9}" src="https://github.com/user-attachments/assets/1d885660-abd4-4871-b6a8-202a9c58c28b" />
<img width="1220" height="756" alt="{EA0C3137-1118-420F-90F8-AC8D2B5324EF}" src="https://github.com/user-attachments/assets/5bda28b4-1342-4bca-8413-9671eebd6235" />
<img width="1238" height="741" alt="{A0C9D028-994C-4514-8AEC-0C2FD297CB7F}" src="https://github.com/user-attachments/assets/17dfcf25-91bb-44a6-a148-17f7e887133c" />
<img width="1220" height="744" alt="{EEBA70FF-7E12-4E61-BB98-A2C3285177EA}" src="https://github.com/user-attachments/assets/dea991b7-dbc8-4843-a047-01ed1650f019" />
<img width="1229" height="745" alt="{46F96A81-11EB-493C-ACC3-A9364A54DDA9}" src="https://github.com/user-attachments/assets/14fb6976-d93b-47cf-bcef-5929e414e6aa" />
<img width="1223" height="749" alt="{789CAAD6-6AEE-45C7-8C8C-39DD1ED4E22C}" src="https://github.com/user-attachments/assets/0882f1dd-44e2-4a15-82e8-92f262903c5a" />



📎 References
McCarthy, C., & Barnes, N. (2012). A Unified Strategy for Landing and Docking Using Spherical Flow Divergence
