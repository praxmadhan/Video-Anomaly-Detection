import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import winsound
import os
from datetime import datetime
from twilio.rest import Client
from twilio_config import ACCOUNT_SID, AUTH_TOKEN, FROM_NUMBER, TO_NUMBER


# ===============================
# Create snapshot folder
# ===============================
if not os.path.exists("snapshots"):
    os.makedirs("snapshots")


# ===============================
# Twilio (SMS) – SAFE INIT
# ===============================
client = None
sms_enabled = True

try:
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
except Exception as e:
    print("⚠️ SMS disabled:", e)
    sms_enabled = False


# ===============================
# Load Models
# ===============================
# CPU-friendly YOLO (stable)
yolo = YOLO("yolov8m.pt")

autoencoder = tf.keras.models.load_model(
    "model/anomaly_model.h5",
    compile=False
)


# ===============================
# Camera
# ===============================
cap = cv2.VideoCapture(1)   # change to 0 if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ===============================
# Calibration (Autoencoder)
# ===============================
CALIB_FRAMES = 120
errors = []
ready = False
threshold = 0

print("Calibration started... Sit still normally...")


# ===============================
# Variables
# ===============================
prev_gray = None

# Camera tamper
tamper_count = 0
TAMPER_FRAMES = 4

# Alarm + SMS control
alarm_cooldown = 0
snapshot_taken = False
anomaly_count = 0
ALERT_THRESHOLD = 5   # continuous anomaly frames
sms_sent = False


# ===============================
# Main Loop
# ===============================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # -------------------------------
    # Gray frame
    # -------------------------------
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # -------------------------------
    # Camera Tamper Detection
    # -------------------------------
    brightness = np.mean(gray_full)
    blur = cv2.Laplacian(gray_full, cv2.CV_64F).var()

    camera_tamper = False
    if brightness < 40 or blur < 40:
        tamper_count += 1
    else:
        tamper_count = 0

    if tamper_count >= TAMPER_FRAMES:
        camera_tamper = True


    # -------------------------------
    # Motion Detection
    # -------------------------------
    motion_score = 0
    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray_full)
        motion_score = np.mean(diff)
    prev_gray = gray_full.copy()


    # -------------------------------
    # YOLO Detection (Weapons / Mask)
    # -------------------------------
    results = yolo(frame, conf=0.35)
    danger = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = yolo.names[cls]
            conf = float(box.conf[0])

            if name in ["knife", "gun", "scissors", "mask"] and conf > 0.45:
                danger = True


    # -------------------------------
    # Autoencoder
    # -------------------------------
    gray = cv2.resize(gray_full, (128, 128))
    gray = gray / 255.0
    gray = gray.reshape(1, 128, 128, 1)

    recon = autoencoder.predict(gray, verbose=0)
    error = np.mean((gray - recon) ** 2)


    # -------------------------------
    # Calibration Phase
    # -------------------------------
    if not ready:

        errors.append(error)

        cv2.putText(
            frame,
            "CALIBRATING... SIT STILL",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        if len(errors) >= CALIB_FRAMES:
            threshold = np.percentile(errors, 97)
            ready = True
            print("Calibration Done | Threshold:", threshold)

        cv2.imshow("Smart Bank Security - Maddy", frame)

        if cv2.waitKey(1) == 27:
            break

        continue


    # -------------------------------
    # Decision System
    # -------------------------------
    anomaly_score = 0

    # Autoencoder
    if error > threshold:
        anomaly_score += 1

    # Motion (relaxed)
    if motion_score > 25 and not danger:
        anomaly_score += 1

    # Weapon / Mask
    if danger:
        anomaly_score += 2

    # Camera Tamper
    if camera_tamper:
        anomaly_score += 2


    status = "NORMAL"
    color = (0, 255, 0)

    if anomaly_score >= 2:

        status = "ANOMALY"
        color = (0, 0, 255)

        if camera_tamper:
            status = "CAMERA TAMPERED"

        anomaly_count += 1

        # Beep (cooldown)
        if alarm_cooldown == 0:
            winsound.Beep(1200, 400)
            alarm_cooldown = 20

        # Snapshot (once per event)
        if not snapshot_taken:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshots/anomaly_{now}.jpg"
            cv2.imwrite(filename, frame)
            print("Snapshot saved:", filename)
            snapshot_taken = True

        # SMS alert (continuous anomaly, SAFE)
        if anomaly_count >= ALERT_THRESHOLD and not sms_sent and sms_enabled:
            try:
                client.messages.create(
                    body="⚠️ ALERT: Continuous anomaly detected in Bank Locker Camera!",
                    from_=FROM_NUMBER,
                    to=TO_NUMBER
                )
                print("SMS alert sent!")
                sms_sent = True
            except Exception as e:
                print("SMS failed:", e)

    else:
        anomaly_count = 0
        sms_sent = False
        snapshot_taken = False


    if alarm_cooldown > 0:
        alarm_cooldown -= 1


    # -------------------------------
    # Display
    # -------------------------------
    info = f"{status} | Err:{round(error,4)} | Motion:{int(motion_score)}"

    cv2.putText(
        frame,
        info,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Smart Bank Security - Maddy", frame)

    if cv2.waitKey(1) == 27:
        break


# ===============================
# Exit
# ===============================
cap.release()
cv2.destroyAllWindows()
print("System Closed")
