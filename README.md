# HygieneVision

This repository implements a real-time computer vision monitoring system designed to enforce environment hygiene compliance. Utilizing a custom-trained YOLOv8 model and OpenCV, the system detects critical adherence failures (e.g., missing hair caps, missing gloves) via webcam streams, recorded video, or static evaluation.

It features a modular inference engine, a FastAPI integration for stateless API consumption, and a Streamlit dashboard with persistent PostgreSQL/MongoDB event logging.

## Project Architecture

* **`hygiene_engine.py`**: The core analytical engine. Encapsulates the YOLOv8 model loading and inference logic. Computes custom Intersection-over-Face (IoF) overlap metrics against secondary Haar Cascade datasets to accurately isolate spatial non-compliance (synthesizing predictions for occluded or heavily bounded classifications).
* **`dashboard.py`**: A synchronized, single-page web application built with Streamlit. Manages local hardware interfacing (webcam) without WebRTC overhead, handles concurrent multi-modal uploads, and provides a real-time reactive UI.
* **`hair-test.py`**: A headless, high-efficiency Python script optimized for edge-device deployment. It removes frontend rendering overhead to strictly execute localized OpenCV operations and console logging.
* **`app.py`**: A FastAPI application providing a REST interface (`/predict`). Returns bounded image arrays natively encoded in Base64 alongside raw class-confidence dictionaries.

## Prerequisites

* Python 3.10+
* MongoDB instance (locally hosted or URI string)
* `pip` / `pyenv`

## Installation

1. Clone this repository and navigate to the root directory.
2. Install the required dependencies:
```bash
pip install -r requirement.txt
pip install pymongo
```

## System Configuration & Execution

The system supports multiple execution vectors depending on the deployment environment.

### 1. Persistent Data Layer (MongoDB)
The analytics dashboard natively writes compliance violations to a local MongoDB instance. 
Start a localized `mongod` background task bound to port `27019` to prevent conflicts with standard operational databases:
```bash
mongod --dbpath ./data_27019 --port 27019 --logpath ./data_27019/mongod.log &
```

### 2. Interactive Dashboard (Streamlit)
For continuous monitoring via local peripherals with full database synchronization:
```bash
streamlit run dashboard.py
```
*Access via: `http://localhost:8501`*

### 3. Edge-Device Headless Script
For maximum compute efficiency and frame processing throughput (e.g., deployment on SOC infrastructure without a GUI/X-server):
```bash
python hair-test.py
```

### 4. REST API (FastAPI)
To expose the inference engine to external services, trigger the Uvicorn ASGI server:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
*API documentation available at: `http://127.0.0.1:8000/docs`*

## Technical Notes
- **Debounce Logic:** The `dashboard.py` state machine implements a 10-second structural debounce per staff identity to prevent database fragmentation during continuous exposure to single-event violations.
- **Bounding Box Heuristics:** The system utilizes coordinate scaling (`minSize`) and aggressive `minNeighbors` configurations within the auxiliary face detection cascade to mitigate dense background artifacts.
