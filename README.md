# AI Motion Detector & Logger 📹

A Python-based real-time motion detection system using OpenCV. It logs motion events, saves snapshots when motion is detected, and records a video that includes only motion-activated frames.

---

## 🚀 Features

- 📷 Real-time motion detection using webcam
- 📝 Logs motion start time, end time, and duration into a CSV file
- 🎥 Records only motion-related video (motion_record.avi)
- 📸 Saves snapshots when motion starts
- 🧠 Filters out false motion using area and aspect ratio thresholds

---

## 📁 Output Files

- motion_log.csv → Start time, end time, and duration of each motion event  
- motion_record.avi → Video with only motion events  
- snapshots/ folder → Images captured at the start of each motion

---

