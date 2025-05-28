import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from threading import Thread, Lock

# ========== CONFIG ==========
IP_CAM_URL = 'http://192.0.0.8:8080/video'
FRAME_SCALE = 0.25
PROCESS_EVERY_N_FRAMES = 10
LOG_INTERVAL = 5  # seconds
LOG_FILE = 'Surveillance_Log.csv'
# ============================

# Load known images
path = 'Training_images'
images = []
classNames = []
for file in os.listdir(path):
    img_path = os.path.join(path, file)
    img = cv2.imread(img_path)
    if img is not None:
        images.append(img)
        classNames.append(os.path.splitext(file)[0].upper())

def findEncodings(images):
    encodeList = []
    for img in images:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_img)
        if encodings:
            encodeList.append(encodings[0])
    return encodeList

def logDetection(name):
    now = datetime.now()
    with open(LOG_FILE, 'a') as f:
        f.write(f"{name},{now.strftime('%Y-%m-%d')},{now.strftime('%H:%M:%S')}\n")
    print(f"[LOG] {name} seen at {now.strftime('%H:%M:%S')}")

# Initialize log file
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w') as f:
        f.write("Name,Date,Time\n")

print("[INFO] Encoding known faces...")
encodeListKnown = findEncodings(images)
print("[INFO] Encoding complete.")

# ==========================
# Threaded Frame Reader
# ==========================
class FrameReader(Thread):
    def __init__(self, src):
        super().__init__()
        self.capture = cv2.VideoCapture(src)
        self.ret = False
        self.frame = None
        self.lock = Lock()
        self.running = True

    def run(self):
        while self.running:
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                self.running = False

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        self.capture.release()

# ==========================
# Main Logic
# ==========================
frame_reader = FrameReader(IP_CAM_URL)
frame_reader.start()

frame_count = 0
last_logged_time = {}

try:
    while True:
        success, img = frame_reader.read()
        if not success or img is None:
            continue

        small_img = cv2.resize(img, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            facesCurFrame = face_recognition.face_locations(rgb_small_img)
            encodesCurFrame = face_recognition.face_encodings(rgb_small_img, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                min_distance = np.min(faceDis)
                if min_distance < 0.6:
                    matchIndex = np.argmin(faceDis)
                    name = classNames[matchIndex]
                    now = datetime.now()

                    if name not in last_logged_time or (now - last_logged_time[name]).total_seconds() > LOG_INTERVAL:
                        logDetection(name)
                        last_logged_time[name] = now

                    y1, x2, y2, x1 = [v * int(1 / FRAME_SCALE) for v in faceLoc]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        frame_count += 1
        cv2.imshow("Surveillance Feed", img)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            print("[INFO] Exiting...")
            break

finally:
    frame_reader.stop()
    frame_reader.join()
    cv2.destroyAllWindows()
