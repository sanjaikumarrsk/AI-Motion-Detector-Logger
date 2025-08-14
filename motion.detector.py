import cv2
import pandas as pd
from datetime import datetime
import numpy as np
import os
import time

if not os.path.exists("snapshots"):
    os.makedirs("snapshots")


video = cv2.VideoCapture(0)


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("motion_record.avi", fourcc, 20.0, (640, 480))

first_frame = None
status_list = [0, 0]
motion_times = []
durations = []
frame_counter = 0
REFRESH_EVERY_N_FRAMES = 120

while True:
    check, frame = video.read()
    status = 0
    frame_counter += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None or frame_counter % REFRESH_EVERY_N_FRAMES == 0:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(delta_frame, 35, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    contours, _ = cv2.findContours(thresh_frame.copy(),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 8000:
            continue
        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = float(w) / h
        if aspect_ratio < 0.2 or aspect_ratio > 2:
            continue

        status = 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    status_list.append(status)
    status_list = status_list[-2:]

    
    if status_list[-1] == 1 and status_list[-2] == 0:
        motion_start_time = datetime.now()
        motion_times.append(motion_start_time)

    
        filename = motion_start_time.strftime("snapshots/motion_%Y%m%d_%H%M%S.jpg")
        cv2.imwrite(filename, frame)


    if status_list[-1] == 0 and status_list[-2] == 1:
        motion_end_time = datetime.now()
        motion_times.append(motion_end_time)

    
        duration = (motion_end_time - motion_start_time).total_seconds()
        durations.append(duration)


    if status == 1:
        out.write(frame)

    motion_text = "Motion Detected" if status == 1 else "No Motion"
    color = (0, 0, 255) if status == 1 else (0, 255, 0)
    cv2.putText(frame, motion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 3)

    time_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, time_text, (10, 460), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

    cv2.imshow("AI Motion Detector", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        if status == 1:
            motion_times.append(datetime.now())
            durations.append((motion_times[-1] - motion_times[-2]).total_seconds())
        break

video.release()
out.release()
cv2.destroyAllWindows()

if len(motion_times) % 2 != 0:
    motion_times.append(datetime.now())
    durations.append((motion_times[-1] - motion_times[-2]).total_seconds())

motion_log = pd.DataFrame(columns=["Start Time", "End Time", "Duration (s)"])
for i in range(0, len(motion_times), 2):
    motion_log = pd.concat([motion_log, pd.DataFrame({
        "Start Time": [motion_times[i].strftime("%Y-%m-%d %H:%M:%S")],
        "End Time": [motion_times[i+1].strftime("%Y-%m-%d %H:%M:%S")],
        "Duration (s)": [round(durations[i//2], 2)]
    })], ignore_index=True)

motion_log.to_csv("motion_log.csv", index=False)
print("✅ Motion log saved as 'motion_log.csv'")
print("✅ Video saved as 'motion_record.avi'")
print("✅ Snapshots saved in 'snapshots/' folder")