import streamlit as st
import cv2
import threading
import time
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import Dict, Any

from smart_classroom.inference import OnnxClassifier
from smart_classroom.ac_controller import ACController


st.set_page_config(page_title="Smart Classroom Edge", layout="wide")

MODEL_DEFAULT = "model.onnx"
CLASS_MAP = {"low": 0, "medium": 1, "high": 2}


def sidebar_controls():
    st.sidebar.header("Settings")
    model_path = st.sidebar.text_input("ONNX model path", value=MODEL_DEFAULT)
    cam_index = st.sidebar.number_input("Camera index", min_value=0, max_value=10, value=0)
    start_button = st.sidebar.button("Start")
    stop_button = st.sidebar.button("Stop")
    return model_path, cam_index, start_button, stop_button


def init_state():
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'frame' not in st.session_state:
        st.session_state.frame = None
    if 'label' not in st.session_state:
        st.session_state.label = 'N/A'
    if 'confidences' not in st.session_state:
        st.session_state.confidences = {'low':0.0,'medium':0.0,'high':0.0}
    if 'history' not in st.session_state:
        st.session_state.history = pd.DataFrame(columns=['timestamp','level','low','medium','high'])
    if 'events' not in st.session_state:
        st.session_state.events = []
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'lock' not in st.session_state:
        st.session_state.lock = threading.Lock()
    if 'ac' not in st.session_state:
        st.session_state.ac = ACController()
    if 'clf' not in st.session_state:
        st.session_state.clf = None


def camera_thread(cam_index: int):
    cap = cv2.VideoCapture(int(cam_index))
    if not cap.isOpened():
        st.session_state.running = False
        return
    clf = st.session_state.clf
    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            # run inference
            try:
                label, confidences = clf.predict(frame)
            except Exception as e:
                label = 'error'
                confidences = {'low':0.0,'medium':0.0,'high':0.0}
            with st.session_state.lock:
                st.session_state.frame = frame.copy()
                st.session_state.label = label
                st.session_state.confidences = confidences
                # append history
                ts = datetime.utcnow()
                row = {'timestamp': ts, 'level': label, 'low': confidences.get('low',0.0), 'medium': confidences.get('medium',0.0), 'high': confidences.get('high',0.0)}
                st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([row])], ignore_index=True)
                # update AC
                event = st.session_state.ac.update(label, confidences)
                st.session_state.events.append(event)
            time.sleep(0.2)
    finally:
        cap.release()


def render_dashboard():
    # layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Live Camera Feed")
        img_placeholder = st.empty()

        st.markdown("---")
        st.header("Occupancy History")
        hist_placeholder = st.empty()

    with col2:
        st.header("Status")
        level_placeholder = st.empty()
        conf_placeholder = st.empty()
        ac_placeholder = st.empty()
        temp_placeholder = st.empty()
        runtime_placeholder = st.empty()

        st.markdown("---")
        st.header("Event Log")
        event_placeholder = st.empty()

    return {
        'img': img_placeholder,
        'hist': hist_placeholder,
        'level': level_placeholder,
        'conf': conf_placeholder,
        'ac': ac_placeholder,
        'temp': temp_placeholder,
        'runtime': runtime_placeholder,
        'events': event_placeholder,
    }


def main():
    model_path, cam_index, start_button, stop_button = sidebar_controls()
    init_state()

    placeholders = render_dashboard()

    # start/stop handling
    if start_button and not st.session_state.running:
        try:
            st.session_state.clf = OnnxClassifier(model_path)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return
        st.session_state.running = True
        st.session_state.start_time = datetime.utcnow()
        # clear history and events if desired
        st.session_state.history = pd.DataFrame(columns=['timestamp','level','low','medium','high'])
        st.session_state.events = []
        t = threading.Thread(target=camera_thread, args=(cam_index,), daemon=True)
        t.start()

    if stop_button and st.session_state.running:
        st.session_state.running = False

    # UI update loop (Streamlit reruns every interaction)
    # show image
    if st.session_state.frame is not None:
        frame = st.session_state.frame
        # convert BGR to RGB for display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholders['img'].image(rgb, channels='RGB', use_column_width=True)
    else:
        placeholders['img'].text("Camera not running")

    # status cards
    placeholders['level'].subheader("Occupancy Level")
    placeholders['level'].write(f"**{st.session_state.label}**")

    placeholders['conf'].subheader("Confidence")
    conf = st.session_state.confidences
    placeholders['conf'].write(pd.DataFrame([conf]).T.rename(columns={0: 'confidence'}))

    ac_status = st.session_state.ac.current_status()
    placeholders['ac'].subheader("AC Status")
    placeholders['ac'].write("ON" if ac_status.on else "OFF")

    placeholders['temp'].subheader("Temperature")
    placeholders['temp'].write(f"{ac_status.temperature} °C" if ac_status.temperature is not None else "—")

    # runtime
    placeholders['runtime'].subheader("Runtime")
    if st.session_state.start_time:
        delta = datetime.utcnow() - st.session_state.start_time
        placeholders['runtime'].write(str(delta).split('.')[0])
    else:
        placeholders['runtime'].write("00:00:00")

    # history chart
    if not st.session_state.history.empty:
        df = st.session_state.history.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        fig = px.line(df, x='timestamp', y=['low','medium','high'], labels={'value':'confidence','variable':'class'})
        placeholders['hist'].plotly_chart(fig, use_container_width=True)
    else:
        placeholders['hist'].text("No history yet")

    # event log
    if st.session_state.events:
        rows = []
        for e in st.session_state.events[-100:][::-1]:
            rows.append({'timestamp': e.timestamp, 'level': e.level, 'message': e.message,
                         'low': e.confidences.get('low',0.0), 'medium': e.confidences.get('medium',0.0), 'high': e.confidences.get('high',0.0)})
        ev_df = pd.DataFrame(rows)
        placeholders['events'].dataframe(ev_df)
    else:
        placeholders['events'].text("No events yet")


if __name__ == '__main__':
    main()
