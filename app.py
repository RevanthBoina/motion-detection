import cv2
import numpy as np
import time
import os
import sqlite3
import threading
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

# Config
DB_PATH = 'database.db'
SNAPSHOTS_DIR = os.path.join('static', 'snapshots')
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

# Global variables state
motion_count = 0
current_motion_state = False
detection_active = True
min_area = 1200
show_mask = False
last_alert_time = 0
BEEP_INTERVAL = 10

latest_frame = None
latest_mask = None

# Initialize Database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS motion_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

def log_to_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO motion_logs (timestamp) VALUES (?)", (timestamp,))
    conn.commit()
    conn.close()

# Background thread for camera processing
def process_camera():
    global latest_frame, latest_mask, motion_count, current_motion_state
    global detection_active, min_area, last_alert_time
    
    cap = cv2.VideoCapture(0)
    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    
    while True:
        if not detection_active:
            # If detection is off, we still want to read frames to show video?
            # Actually, the user asked to toggle detection. Let's just bypass detection logic
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.1)
                continue
            
            latest_frame = frame.copy()
            latest_mask = np.zeros_like(frame) # Empty mask
            current_motion_state = False
            time.sleep(0.03)
            continue
            
        ret, frame = cap.read()
        if not ret:
            # Reinitialize camera if it drops/ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.1)
            continue
        
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
            # Draw green bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        current_motion_state = motion
        
        if motion:
            now = time.time()
            if now - last_alert_time >= BEEP_INTERVAL:
                motion_count += 1
                last_alert_time = now
                log_to_db()
                img_name = os.path.join(SNAPSHOTS_DIR, f"motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(img_name, frame)
                
        # Draw motion counter on frame
        cv2.putText(frame, f"Motions: {motion_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Convert 1-channel mask to 3-channel so it renders easily
        mask_3ch = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
                    
        # Update thread-safe globals
        latest_frame = frame.copy()
        latest_mask = mask_3ch.copy()
        time.sleep(0.03)

# Start parsing thread
t = threading.Thread(target=process_camera, daemon=True)
t.start()

def generate_video():
    global latest_frame, latest_mask, show_mask
    while True:
        frame_to_show = latest_mask if show_mask else latest_frame
        
        if frame_to_show is None:
            time.sleep(0.1)
            continue
            
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame_to_show)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def status():
    return jsonify({
        'motion': current_motion_state,
        'count': motion_count
    })

@app.route('/api/logs')
def logs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp FROM motion_logs ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return jsonify([row[0] for row in rows])

@app.route('/api/snapshots')
def snapshots():
    if not os.path.exists(SNAPSHOTS_DIR):
        return jsonify([])
    files = os.listdir(SNAPSHOTS_DIR)
    # Filter jpg
    files = [f for f in files if f.endswith('.jpg')]
    # Sort by descending order
    files.sort(reverse=True)
    # Return top 20
    return jsonify(files[:20])

@app.route('/api/config', methods=['POST'])
def config():
    global detection_active, min_area, show_mask
    data = request.json
    
    if 'detection_active' in data:
        detection_active = bool(data['detection_active'])
    if 'min_area' in data:
        min_area = int(data['min_area'])
    if 'show_mask' in data:
        show_mask = bool(data['show_mask'])
        
    return jsonify({
        'status': 'success',
        'detection_active': detection_active,
        'min_area': min_area,
        'show_mask': show_mask
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
