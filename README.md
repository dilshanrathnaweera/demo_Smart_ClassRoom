# Smart Classroom Edge AI System

Production-ready prototype for occupancy-level inference, AC simulation, and Streamlit dashboard.

## Features
- Loads Azure Custom Vision ONNX model (input tensor `data`, 224x224 RGB)
- Real-time webcam capture and continuous prediction
- AC simulation mapping occupancy levels to AC state and temperature
- Streamlit dashboard with live feed, runtime, history graph (Plotly), and event log
- Docker support

## Quickstart (local)

1. Create and activate a Python venv

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Unix
source venv/bin/activate
```

2. Install deps

```bash
pip install -r requirements.txt
```

3. Run Streamlit

```bash
streamlit run smart_classroom/dashboard.py --server.port 8501
```

4. In the app, set the path to your ONNX model and start the camera.

## Docker

Build:

```bash
docker build -t smart-classroom-edge:latest .
```

Run (Linux with webcam):

```bash
docker run --rm -it --device=/dev/video0:/dev/video0 -p 8501:8501 smart-classroom-edge:latest
```

On Windows, pass through camera differently (e.g., use Host networking or USB passthrough).

## Notes
- The app expects an ONNX model whose input tensor name is `data` and outputs class scores for `low`, `medium`, `high`.
- If your model uses a different preprocessing (mean/std or BGR order), update `smart_classroom/inference.py` accordingly.
