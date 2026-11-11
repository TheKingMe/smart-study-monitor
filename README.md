# Smart Study Monitor 🎓👁️📱 [Enhanced & Stabilized Edition]

An AI-powered computer vision productivity monitor that detects drowsiness, face occlusion, and phone distractions in real time using **OpenCV**, **MediaPipe Face Mesh (`cvzone`)**, and **YOLOv8**.

---

## 🚀 Key Stability & Detection Improvements

1. **Multi-Threaded Asynchronous YOLO Detection**:
   - Phone detection runs in a dedicated background worker thread (`AsyncPhoneDetector`).
   - The main camera feed and face tracking run at a buttery-smooth **30 FPS** with **zero camera latency or stutter**.

2. **6-Point Biometric Eye Aspect Ratio (EAR)**:
   - Uses a full 6-point dual-eye vertical-to-horizontal distance formulation (`160-144`, `158-153` over `33-133` on the left; `385-380`, `387-373` over `362-263` on the right).
   - Exponential Moving Average (EMA) smoothing eliminates flickering numbers caused by sensor noise or micro-twitches.

3. **Natural Blink Filtering & State Hysteresis**:
   - Drowsiness requires persistent eye closure (`EAR < 15`) for ~0.75 seconds to ensure natural blinks (150–250ms) do not trigger false alarms.
   - Hysteresis recovery (`EAR > 19`) prevents rapid flickering when eyes are opening or closing.

4. **DirectShow (`CAP_DSHOW`) Windows Camera Backend**:
   - Prevents Windows Media Foundation (MSMF) buffer lag and sample drop errors.
   - Buffer size set to 1 for zero-latency real-time video playback.

5. **Smooth Bounding Box Tracking (LERP)**:
   - The phone bounding box interpolates smoothly across frames and maintains a 0.45s persistence window, eliminating box flickering.

---

## 📁 Project Structure

```text
Study code/
├── app.py               # Enhanced real-time monitoring application
├── run.bat              # One-click Windows launcher
├── requirements.txt     # Python dependencies
├── alarm.mp3            # Audio alert for drowsiness ("Wake up and study!")
├── faudio.mp3           # Audio alert for face covered ("Don't cover your face!")
├── paudio.mp3           # Audio alert for phone detection ("Put the phone away!")
├── yolov8n.pt           # YOLOv8 nano model
├── face_landmarker.task # MediaPipe Face Mesh model
└── README.md            # Documentation
```

---

## 🎯 How to Run

Double-click **`run.bat`** or run:
```bash
python app.py
```
*(Press **`q`** or **`ESC`** in the camera window to exit).*
