# Asset Counting and Verification using YOLO on the VPU of the OAK-D Pro Camera

This repository provides the complete implementation of a real-time object detection system for asset counting and verification. The system supports multiple hardware platforms, including the Raspberry Pi 4B CPU, the OAK-D Pro’s VPU (MyriadX), and the Google Coral Edge TPU.

The solution is based on a one-shot object detection strategy that leverages lightweight YOLO models and ORB feature matching to detect assets using a single reference image. The results are visualized through a Flask-based web dashboard that reports detection accuracy, power consumption, temperature, and object counts in real time.

## Project Objectives

- Enable real-time object detection and asset tracking in indoor, GPS-denied environments.
- Deploy and benchmark lightweight models on edge devices.
- Use one-shot object detection to avoid retraining for each new asset class.
- Offload computationally intensive inference to the OAK-D Pro’s VPU or Coral Edge TPU.
- Provide real-time monitoring through a dynamic Flask dashboard.

## Repository Contents

| File Name | Description |
|-----------|-------------|
| `Yolov5_cpu_code.py` | Benchmarks YOLOv5n/s/m/l models on Raspberry Pi 4B CPU. Measures latency, FPS, and power consumption. |
| `Yolov5_vpu_code.py` | Final deployment script for running YOLOv5n on the OAK-D Pro’s VPU. Includes ORB-based one-shot object verification and dashboard integration. |
| `Yolov8_cpu_code.py` | Evaluates YOLOv8n, YOLOv8s, and YOLOv8m models on Raspberry Pi CPU. Includes real-time ORB matching and statistical tracking. |
| `Yolo_tpu_code.py` | Implements bottle detection using a YOLO-based model on the Coral Edge TPU. Performs reference matching using PyCoral utilities. |
| `yolov5n_openvino_2022_1_6shave_1.blob` | Precompiled OpenVINO `.blob` model for YOLOv5n, optimized for real-time execution on the OAK-D Pro VPU. |
| `dashboard.html` | Front-end HTML template used by the Flask server to display detection stats, temperature, and power consumption in real time. |
| `README.md` | This documentation file providing an overview of the codebase and project structure. |

## Detection Strategy

This project adopts a one-shot detection strategy rather than conventional supervised training. Object identification is performed using a single reference image and feature-based similarity, which significantly improves flexibility and deployability in dynamic environments such as warehouses.

Key advantages of this method include:
- No need for large-scale retraining.
- Detection and classification using only one image per object.
- Immediate scalability to new asset types.

## Dashboard Integration

The system includes a real-time web dashboard developed using Flask and SocketIO. It visualizes the following performance metrics:
- Asset count and detection confidence
- Frame processing time and FPS
- CPU usage, power consumption, and thermal load
- One-shot match verification using ORB descriptors

The dashboard supports cross-platform operation and can be accessed via any modern web browser.

## Licensing and Use

This repository is intended for academic, educational, and research purposes. Please contact the author for licensing questions or reuse in production environments.

