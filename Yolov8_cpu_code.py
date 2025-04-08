import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import time
import os
import depthai as dai
from ultralytics import YOLO
import psutil
 
# === Flask Setup ===
app = Flask(__name__, template_folder='templates', static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*")
 
# === YOLOv8 Model ===
model = YOLO("yolov8m.pt")  # Use any YOLOv8 model
 
# === Load Reference Image for One-Shot Matching ===
reference_image = cv2.imread("reference_bottle.jpg", cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create(nfeatures=1000)
ref_kp, ref_des = orb.detectAndCompute(reference_image, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
 
def is_similar_to_reference(crop_rgb):
    try:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        kp2, des2 = orb.detectAndCompute(gray, None)
        if des2 is None or ref_des is None:
            return False
        matches = matcher.match(ref_des, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        score = sum([m.distance for m in matches[:10]]) / 10
        print(f"[MATCH SCORE]: {score}")
        return score < 45  # Adjust threshold as needed
    except Exception as e:
        print(f"[MATCH ERROR]: {e}")
        return False
 
# === DepthAI Setup ===
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(640, 480)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
 
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam.preview.link(xout.input)
 
# === State Tracking ===
tracked_objects = {}
MAX_LIFESPAN = 30
bounding_boxes = []
total_bottle_count = 0
avg_confidence = 0.0
frame_times, confidence_scores, fps_values = [], [], []
power_values = []
 
# === Matplotlib Plots ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))
image_display = ax1.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
ax1.axis('off')
ax1.set_title("One-Shot Reference Detection (ORB + YOLOv8)")
 
power_plot, = ax2.plot([], [], label="Power (W)", color="yellow")
ax2.set_title("Estimated Power Consumption")
ax2.set_xlabel("Frame")
ax2.set_ylabel("Watts")
ax2.grid(True)
ax2.legend()
 
@app.route("/")
def index():
    return render_template("dashboard.html")
 
def send_bottle_stats():
    global total_bottle_count, avg_confidence, frame_times, fps_values
    while True:
        avg_speed = round(sum(frame_times) / len(frame_times), 2) if frame_times else 0.0
        avg_conf = round(avg_confidence * 100, 2)
        avg_fps = round(sum(fps_values) / len(fps_values), 2) if fps_values else 0.0
        recent_detections = sum(1 for f in confidence_scores[-10:] if f > 0.6)
 
        cpu_usage = psutil.cpu_percent(interval=None)
        voltage = 5.0
        scaling_factor = 3.5
        power_watts = scaling_factor * (cpu_usage / 100) * voltage * psutil.cpu_count(logical=False)
 
        socketio.emit("bottle_stats", {
            "count": total_bottle_count,
            "accuracy": avg_conf,
            "speed": avg_speed,
            "fps": avg_fps,
            "recent_hits": recent_detections,
            "power": round(cpu_usage, 1),
            "energy": {
                "power_watts": round(power_watts, 2)
            }
        })
        time.sleep(1)
 
def update_frame(_):
    global total_bottle_count, avg_confidence, frame_times, fps_values, confidence_scores, power_values
    start = time.time()
 
    try:
        in_frame = q.get()
        frame = in_frame.getCvFrame()
    except:
        return image_display
 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    for b in bounding_boxes:
        b[0].remove()
        b[1].remove()
    bounding_boxes.clear()
 
    expired = [k for k, (_, life, _) in tracked_objects.items() if life - 1 <= 0]
    for k in expired:
        del tracked_objects[k]
    for k in tracked_objects:
        box, life, center = tracked_objects[k]
        tracked_objects[k] = (box, life - 1, center)
 
    results = model(rgb, verbose=False)
    frame_confidences = []
 
    for r in results:
        for box in r.boxes:
            conf = float(box.conf.item())
            if conf < 0.6:
                continue
 
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop_img = rgb[y1:y2, x1:x2]
 
            if not is_similar_to_reference(crop_img):
                continue
 
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            current_box = (x1, y1, x2, y2)
            frame_confidences.append(conf)
 
            is_new = True
            for _, (prev_box, _, _) in tracked_objects.items():
                if compute_iou(current_box, prev_box) > 0.3:
                    is_new = False
                    break
 
            if is_new:
                tracked_objects[f"{cx}_{cy}_{time.time()}"] = (current_box, MAX_LIFESPAN, (cx, cy))
                total_bottle_count += 1
 
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor="lime", facecolor="none")
            label = ax1.text(x1, y1 - 5, f"Match ({conf:.2f})", color="white",
                             fontsize=8, bbox=dict(facecolor="green", alpha=0.6))
            ax1.add_patch(rect)
            bounding_boxes.append((rect, label))
 
    if frame_confidences:
        confidence_scores += frame_confidences
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
 
    duration = time.time() - start
    frame_times.append(duration * 1000)
    fps_values.append(1.0 / duration if duration > 0 else 0)
 
    if len(frame_times) > 30:
        frame_times.pop(0)
    if len(fps_values) > 30:
        fps_values.pop(0)
 
    cpu_usage = psutil.cpu_percent(interval=None)
    voltage = 5.0
    scaling_factor = 3.5
    power_watts = scaling_factor * (cpu_usage / 100) * voltage * psutil.cpu_count(logical=False)
    power_values.append(power_watts)
    if len(power_values) > 50:
        power_values.pop(0)
 
    power_plot.set_data(range(len(power_values)), power_values)
    ax2.set_xlim(0, len(power_values))
    ax2.set_ylim(0, max(power_values + [10]))
 
    image_display.set_data(rgb)
    return image_display
 
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2
    inter_x1 = max(x1, x1p)
    inter_y1 = max(y1, y1p)
    inter_x2 = min(x2, x2p)
    inter_y2 = min(y2, y2p)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2p - x1p) * (y2p - y1p)
    return inter_area / float(area1 + area2 - inter_area + 1e-5)
 
if __name__ == "__main__":
    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="video", maxSize=4, blocking=False)
 
        threading.Thread(target=lambda: socketio.run(app, host="0.0.0.0", port=5000), daemon=True).start()
        threading.Thread(target=send_bottle_stats, daemon=True).start()
 
        ani = animation.FuncAnimation(fig, update_frame, interval=50, blit=False)
        plt.tight_layout()
        plt.show()