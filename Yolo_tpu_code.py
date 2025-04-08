import os
import cv2
import time
import numpy as np
import threading
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from flask import Flask, render_template
from flask_socketio import SocketIO

from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.detect import get_objects
from pycoral.adapters.common import input_size
from PIL import Image

# === Flask Setup ===
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# === Load Edge TPU model ===
interpreter = make_interpreter('model_quant_edgetpu.tflite')
interpreter.allocate_tensors()

# === Get Input Dimensions ===
input_width, input_height = input_size(interpreter)

# === Load Reference Image ===
reference_img_path = "reference.jpg"
reference_img = cv2.imread(reference_img_path)
if reference_img is None:
    raise ValueError("❌ Reference image not found!")

reference_img_rgb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
ref_resized = cv2.resize(reference_img_rgb, (input_width, input_height))
ref_tensor = np.asarray(ref_resized, dtype=np.uint8)

interpreter.set_tensor(interpreter.get_input_details()[0]['index'], [ref_tensor])
interpreter.invoke()
ref_detections = get_objects(interpreter, score_threshold=0.5)

if len(ref_detections) == 0:
    raise ValueError("❌ No object detected in reference image!")

reference_class = ref_detections[0].id
print(f"✅ Reference object class detected: {reference_class}")

# === Load Video ===
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open video!")

# === Matplotlib Setup ===
fig, ax = plt.subplots(figsize=(8, 6))
image_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
ax.set_title(f"Processing {video_path} with EdgeTPU")

bounding_boxes = []
total_bottle_count = 0
tracked_bottles = {}

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
    return inter_area / float(area1 + area2 - inter_area)

def send_bottle_count():
    global total_bottle_count
    while True:
        socketio.emit("bottle_count", {"count": total_bottle_count})
        time.sleep(1)

def update_frame(_):
    global total_bottle_count
    success, frame = cap.read()
    if not success:
        cap.release()
        return image_display

    frame_resized = cv2.resize(frame, (input_width, input_height))
    input_tensor = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], [input_tensor])
    interpreter.invoke()
    results = get_objects(interpreter, score_threshold=0.5)

    frame_display = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)

    for box in bounding_boxes:
        box.remove()
    bounding_boxes.clear()

    detected_new_bottle = False

    for obj in results:
        if obj.id != reference_class:
            continue

        x1, y1, x2, y2 = obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax
        scale_x = frame_display.shape[1] / input_width
        scale_y = frame_display.shape[0] / input_height
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

        is_new = True
        for _, prev_box in tracked_bottles.items():
            if compute_iou((x1, y1, x2, y2), prev_box) > 0.7:
                is_new = False
                break

        if is_new:
            tracked_bottles[f"{x1}_{y1}_{x2}_{y2}"] = (x1, y1, x2, y2)
            total_bottle_count += 1
            detected_new_bottle = True

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        bounding_boxes.append(rect)
        ax.text(x1, y1 - 5, f"Bottle ({obj.score:.2f})", color="red", fontsize=10, weight="bold", backgroundcolor="black")

    if detected_new_bottle:
        print(f"📦 New bottle detected. Total: {total_bottle_count}")

    image_display.set_data(frame_rgb)
    return image_display

@app.route("/")
def index():
    return render_template("dashboard.html")

# === Main Entrypoint ===
if __name__ == "__main__":
    threading.Thread(target=lambda: socketio.run(app, host="0.0.0.0", port=5000), daemon=True).start()
    threading.Thread(target=send_bottle_count, daemon=True).start()
    ani = animation.FuncAnimation(fig, update_frame, interval=100)
    plt.show()
