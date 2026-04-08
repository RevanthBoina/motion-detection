import cv2
import numpy as np
import time
import os
from datetime import datetime

# --- NEW: User Selection for Source ---
print("Select Video Source:")
print("1. Live Camera")
print("2. Recorded Video File")
choice = input("Enter choice (1 or 2): ")

if choice == '1':
    VIDEO_SOURCE = 0
else:
    VIDEO_SOURCE = input("Enter the full path to the video file (e.g., video.mp4): ")
# ---------------------------------------

BEEP_INTERVAL = 10
SAVE_DIR = "motion_logs"

os.makedirs(SAVE_DIR, exist_ok=True)

def nothing(x):
    pass

cv2.namedWindow("Controls")
cv2.createTrackbar("Sensitivity", "Controls", 1200, 5000, nothing)

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open source {VIDEO_SOURCE}")
    exit()

bg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=True
)

last_alert_time = 0
motion_count = 0

def log_motion():
    # Use 'a' to append to the file
    with open("motion_log.txt", "a") as f:
        f.write(f"Motion detected at {datetime.now()}\n")

while True:
    ret, frame = cap.read()
    if not ret:
        # If it's a video file, this usually means the video ended
        print("End of video stream or cannot fetch frame.")
        break

    min_area = cv2.getTrackbarPos("Sensitivity", "Controls")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    fg = bg.apply(blur)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion = False

    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        motion = True
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if motion:
        if time.time() - last_alert_time >= BEEP_INTERVAL:
            motion_count += 1
            last_alert_time = time.time()
            log_motion()

            img_name = f"{SAVE_DIR}/motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(img_name, frame)

    cv2.putText(frame, f"Motions: {motion_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Motion Detection", frame)
    cv2.imshow("Mask", fg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()