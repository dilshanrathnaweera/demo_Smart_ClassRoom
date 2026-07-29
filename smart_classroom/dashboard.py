import sys
from pathlib import Path

# Ensure the repository root is on sys.path and add it at the front so imports
# like `import smart_classroom` work when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

import streamlit as st
import queue
import cv2
import threading
import time
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import Dict, Any
import os

from smart_classroom.inference import OnnxClassifier
from smart_classroom.ac_controller import ACController
from smart_classroom.attendance import AttendanceController
from smart_classroom.thumbnail import save_thumbnail


st.set_page_config(page_title="Smart Classroom Edge", layout="wide")

# Inject a minimal dark theme and spacing CSS to give a modern control-center look
_DARK_CSS = """
body { background-color: #0f1720; color: #e6eef3; }
.stApp { background-color: #0f1720; }
.css-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }
.card { background: #0b1320; padding: 12px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.6); }
.small-muted { color: #9fb0c8; font-size: 0.9em }
.large-title { font-size: 1.25rem; font-weight: 700; }
.center { display:flex; align-items:center; justify-content:center }
.thumb-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap:8px }
"""
try:
    st.markdown(f"<style>{_DARK_CSS}</style>", unsafe_allow_html=True)
except Exception:
    pass

MODEL_DEFAULT = "model.onnx"
CLASS_MAP = {"low": 0, "medium": 1, "high": 2}


def sidebar_controls():
    st.sidebar.header("Settings")
    model_path = st.sidebar.text_input("ONNX model path", value=MODEL_DEFAULT)
    cam_index = st.sidebar.number_input("Camera index", min_value=0, max_value=10, value=0)
    source = st.sidebar.radio("Source", ["Live Camera", "Upload Video"])
    # initialize all controls so return values are defined for both branches
    start_button = False
    stop_button = False
    upload_file = None
    upload_start = False

    if source == 'Upload Video':
        upload_file = st.sidebar.file_uploader("Upload video (mp4, avi, mov)", type=['mp4', 'avi', 'mov'])
        upload_start = st.sidebar.button("Start Upload")
        stop_button = st.sidebar.button("Stop")
    else:
        start_button = st.sidebar.button("Start")
        stop_button = st.sidebar.button("Stop")

    return model_path, cam_index, start_button, stop_button, source, upload_file, upload_start


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
    if 'attendance' not in st.session_state:
        # configurable: 8s presence required, 0.5 confidence threshold
        st.session_state.attendance = AttendanceController(required_duration_seconds=8, confidence_threshold=0.5, absence_duration_seconds=3)
    if 'clf' not in st.session_state:
        st.session_state.clf = None
    # storage integration: create Storage and a background writer queue/thread for non-blocking writes
    if 'storage' not in st.session_state:
        try:
            from smart_classroom.storage import Storage
            st.session_state.storage = Storage()
        except Exception:
            st.session_state.storage = None
    if 'storage_write_queue' not in st.session_state:
        st.session_state.storage_write_queue = queue.Queue()

        def _writer_loop(q, storage):
            while True:
                task = q.get()
                if task is None:
                    break
                method, args = task
                try:
                    if storage:
                        getattr(storage, method)(*args)
                except Exception:
                    # don't raise in background writer; could log
                    pass

        t = threading.Thread(target=_writer_loop, args=(st.session_state.storage_write_queue, st.session_state.storage), daemon=True)
        t.start()
        st.session_state._storage_writer_thread = t

    # preload persisted history and events into session state
    try:
        if st.session_state.storage:
            # load recent occupancy history (oldest first)
            oh = st.session_state.storage.get_occupancy_history(limit=1000)[::-1]
            if not oh:
                st.session_state.history = pd.DataFrame(columns=['timestamp','level','low','medium','high'])
            else:
                df = pd.DataFrame([{'timestamp': r['timestamp'], 'level': None, 'low': r['low'], 'medium': r['medium'], 'high': r['high']} for r in oh])
                st.session_state.history = df
            # load recent events
            evs = st.session_state.storage.get_events()
            st.session_state.events = []
            for e in evs[::-1]:
                # keep structure similar to ACEvent minimal fields
                st.session_state.events.append(type('E', (), {'timestamp': e['timestamp'], 'level': e['level'], 'message': e['message'], 'confidences': e['confidences']}))
    except Exception:
        pass


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
                # update AC and record event
                event = st.session_state.ac.update(label, confidences)
                st.session_state.events.append(event)
                # enqueue writes to storage (non-blocking)
                try:
                    q = st.session_state.storage_write_queue
                    # capture thumbnail for this frame/event
                    thumbnail_path = ''
                    try:
                        thumbnail_path = save_thumbnail(frame.copy(), timestamp=event.timestamp)
                    except Exception:
                        thumbnail_path = ''
                    # add event (use ACEvent timestamp)
                    q.put(('add_event', (label, event.message, confidences, event.timestamp, thumbnail_path)))
                    # add occupancy sample
                    q.put(('add_occupancy', (confidences.get('low',0.0), confidences.get('medium',0.0), confidences.get('high',0.0), ts)))
                    # attendance detection
                    try:
                        att = None
                        if 'attendance' in st.session_state:
                            att = st.session_state.attendance.process_sample(confidences, ts)
                        if att is not None:
                            # storage.add_attendance(person, timestamp=None, level=None, confidence=None)
                            q.put(('add_attendance', (att.get('person', 'unknown'), att.get('timestamp'), att.get('level'), att.get('confidence'), thumbnail_path)))
                            # also log an event for UI (include thumbnail)
                            q.put(('add_event', ('medium', f"attendance:{att.get('person')}", {'low':0.0,'medium':att.get('confidence',0.0),'high':0.0}, att.get('timestamp'), thumbnail_path)))
                    except Exception:
                        pass
                except Exception:
                    pass
            time.sleep(0.2)
    finally:
        cap.release()


def process_video_file(path: str, clf, storage=None, storage_queue=None, attendance=None, thumbnail_dir=None, update_callback=None, frame_delay=0.2):
    """Process a video file frame-by-frame, run inference, and persist results.

    - path: filesystem path to video file
    - clf: classifier with .predict(frame) -> (label, confidences)
    - storage: optional Storage instance for synchronous writes
    - storage_queue: optional queue for background writes (takes precedence)
    - attendance: optional AttendanceController
    - thumbnail_dir: optional directory to save thumbnails (pass Path or str)
    - update_callback: optional callable(frame, label, confidences, timestamp)
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                label, confidences = clf.predict(frame)
            except Exception:
                label = 'error'
                confidences = {'low': 0.0, 'medium': 0.0, 'high': 0.0}
            ts = datetime.utcnow()
            # UI update callback (dashboard provides one)
            if update_callback:
                try:
                    update_callback(frame.copy(), label, confidences, ts)
                except Exception:
                    pass

            # persist or enqueue
            thumb_path = ''
            try:
                if thumbnail_dir:
                    from pathlib import Path as _P
                    td = _P(thumbnail_dir)
                    thumb_path = save_thumbnail(frame.copy(), out_dir=td, timestamp=ts)
            except Exception:
                thumb_path = ''

            if storage_queue is not None:
                try:
                    storage_queue.put(('add_event', (label, 'video_frame', confidences, ts, thumb_path)))
                    storage_queue.put(('add_occupancy', (confidences.get('low', 0.0), confidences.get('medium', 0.0), confidences.get('high', 0.0), ts)))
                    if attendance is not None:
                        att = attendance.process_sample(confidences, ts)
                        if att is not None:
                            storage_queue.put(('add_attendance', (att.get('person', 'unknown'), att.get('timestamp'), att.get('level'), att.get('confidence'), thumb_path)))
                            storage_queue.put(('add_event', ('medium', f"attendance:{att.get('person')}", {'low':0.0,'medium':att.get('confidence',0.0),'high':0.0}, att.get('timestamp'), thumb_path)))
                except Exception:
                    pass
            elif storage is not None:
                try:
                    storage.add_event(label, 'video_frame', confidences, ts, thumb_path)
                    storage.add_occupancy(confidences.get('low', 0.0), confidences.get('medium', 0.0), confidences.get('high', 0.0), ts)
                    if attendance is not None:
                        att = attendance.process_sample(confidences, ts)
                        if att is not None:
                            storage.add_attendance(att.get('person', 'unknown'), att.get('timestamp'), att.get('level'), att.get('confidence'), thumb_path)
                            storage.add_event('medium', f"attendance:{att.get('person')}", {'low':0.0,'medium':att.get('confidence',0.0),'high':0.0}, att.get('timestamp'), thumb_path)
                except Exception:
                    pass

            time.sleep(frame_delay)
    finally:
        cap.release()


def render_dashboard():
    # top summary and large occupancy gauge
    c1, c2 = st.columns([2, 1])
    with c1:
        today_att_card = st.empty()
        total_events_card = st.empty()
    with c2:
        current_occ_card = st.empty()

    # layout with tabs: Live / History / Attendance
    tab_live, tab_history, tab_att = st.tabs(["Live", "History", "Attendance"])

    with tab_live:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="large-title">Live Camera Feed</div>', unsafe_allow_html=True)
            img_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="large-title">Occupancy History (recent)</div>', unsafe_allow_html=True)
            hist_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="large-title">Status</div>', unsafe_allow_html=True)
            level_placeholder = st.empty()
            conf_placeholder = st.empty()
            ac_placeholder = st.empty()
            temp_placeholder = st.empty()
            runtime_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="large-title">Event Log</div>', unsafe_allow_html=True)
            event_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_history:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="large-title">Full Occupancy History</div>', unsafe_allow_html=True)
        history_chart = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="large-title">Event Log (persisted)</div>', unsafe_allow_html=True)
        history_events = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_att:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="large-title">Attendance Records</div>', unsafe_allow_html=True)
        fcol1, fcol2, fcol3 = st.columns([1, 2, 1])
        with fcol1:
            date_range = st.date_input("Date range", [])
        with fcol2:
            person_filter = st.text_input("Person (leave empty for all)")
        with fcol3:
            att_search = st.text_input("Search attendance (level/confidence)")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        att_table = st.empty()
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="large-title">Export</div>', unsafe_allow_html=True)
        export_att = st.button("Export attendance CSV")
        att_export_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    return {
        'img': img_placeholder,
        'hist': hist_placeholder,
        'level': level_placeholder,
        'conf': conf_placeholder,
        'ac': ac_placeholder,
        'temp': temp_placeholder,
        'runtime': runtime_placeholder,
        'events': event_placeholder,
        'history_chart': history_chart,
        'history_events': history_events,
        'attendance_table': att_table,
        'attendance_export_placeholder': att_export_placeholder,
        'attendance_export_button': export_att,
        'today_att_card': today_att_card,
        'total_events_card': total_events_card,
        'current_occ_card': current_occ_card,
    }


def main():
    model_path, cam_index, start_button, stop_button, source, upload_file, upload_start = sidebar_controls()
    init_state()

    placeholders = render_dashboard()

    # start/stop handling
    if source == 'Live Camera' and start_button and not st.session_state.running:
        # resolve model path and print diagnostics to sidebar to help NO_SUCHFILE issues
        try:
            raw = model_path
            cwd = os.getcwd()
            abs_path = os.path.abspath(raw)
            candidates = [Path(raw), ROOT / raw, Path(cwd) / raw]
            resolved = None
            for c in candidates:
                try:
                    p = Path(c)
                except Exception:
                    p = None
                if p and p.exists():
                    resolved = p
                    break

            # diagnostics
            try:
                st.sidebar.markdown(f"**Model diagnostics**")
                st.sidebar.text(f"cwd: {cwd}")
                st.sidebar.text(f"raw: {raw}")
                st.sidebar.text(f"abs: {abs_path}")
                st.sidebar.text(f"resolved: {resolved}")
                st.sidebar.text(f"exists: {bool(resolved and resolved.exists())}")
            except Exception:
                pass

            # prefer resolved absolute path when available
            if resolved:
                st.session_state.clf = OnnxClassifier(str(resolved))
            else:
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

    # Upload start handling
    if source == 'Upload Video' and upload_start and upload_file is not None and not st.session_state.get('running_upload', False):
        # write uploaded file to disk and start processing thread
        try:
            from pathlib import Path
            up_dir = Path.cwd() / 'data' / 'uploads'
            up_dir.mkdir(parents=True, exist_ok=True)
            out_path = up_dir / upload_file.name
            # stream bytes to file
            with open(out_path, 'wb') as f:
                f.write(upload_file.getbuffer())
            # ensure classifier loaded
            # resolve model path and diagnostics for upload flow as well
            try:
                raw = model_path
                cwd = os.getcwd()
                abs_path = os.path.abspath(raw)
                candidates = [Path(raw), ROOT / raw, Path(cwd) / raw]
                resolved = None
                for c in candidates:
                    try:
                        p = Path(c)
                    except Exception:
                        p = None
                    if p and p.exists():
                        resolved = p
                        break

                try:
                    st.sidebar.markdown(f"**Model diagnostics**")
                    st.sidebar.text(f"cwd: {cwd}")
                    st.sidebar.text(f"raw: {raw}")
                    st.sidebar.text(f"abs: {abs_path}")
                    st.sidebar.text(f"resolved: {resolved}")
                    st.sidebar.text(f"exists: {bool(resolved and resolved.exists())}")
                except Exception:
                    pass

                if resolved:
                    st.session_state.clf = OnnxClassifier(str(resolved))
                else:
                    st.session_state.clf = OnnxClassifier(model_path)
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return
            st.session_state.running_upload = True
            st.session_state.start_time = datetime.utcnow()

            def _upload_callback(frame, label, confidences, ts):
                # mimic camera_thread updates into session state
                with st.session_state.lock:
                    st.session_state.frame = frame
                    st.session_state.label = label
                    st.session_state.confidences = confidences
                    row = {'timestamp': ts, 'level': label, 'low': confidences.get('low',0.0), 'medium': confidences.get('medium',0.0), 'high': confidences.get('high',0.0)}
                    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([row])], ignore_index=True)
                    event = st.session_state.ac.update(label, confidences)
                    st.session_state.events.append(event)
                    # enqueue writes using storage_write_queue
                    try:
                        q = st.session_state.storage_write_queue
                        thumb = ''
                        try:
                            thumb = save_thumbnail(frame.copy(), timestamp=ts)
                        except Exception:
                            thumb = ''
                        q.put(('add_event', (label, event.message, confidences, ts, thumb)))
                        q.put(('add_occupancy', (confidences.get('low',0.0), confidences.get('medium',0.0), confidences.get('high',0.0), ts)))
                        try:
                            att = st.session_state.attendance.process_sample(confidences, ts)
                            if att is not None:
                                q.put(('add_attendance', (att.get('person','unknown'), att.get('timestamp'), att.get('level'), att.get('confidence'), thumb)))
                                q.put(('add_event', ('medium', f"attendance:{att.get('person')}", {'low':0.0,'medium':att.get('confidence',0.0),'high':0.0}, att.get('timestamp'), thumb)))
                        except Exception:
                            pass
                    except Exception:
                        pass

            # start background thread to process file
            t = threading.Thread(target=process_video_file, args=(str(out_path), st.session_state.clf), kwargs={'storage_queue': st.session_state.storage_write_queue, 'attendance': st.session_state.attendance, 'thumbnail_dir': str(Path.cwd() / 'data' / 'thumbnails'), 'update_callback': _upload_callback}, daemon=True)
            t.start()
            st.session_state._upload_thread = t
        except Exception as e:
            st.error(f"Failed to start upload processing: {e}")

    if (stop_button and st.session_state.running) or (stop_button and st.session_state.get('running_upload', False)):
        st.session_state.running = False
        st.session_state.running_upload = False

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

    # history chart is also available in the History tab; here show a small recent chart
    if not st.session_state.history.empty:
        df = st.session_state.history.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # enhanced area chart for analytics
        fig = px.area(df, x='timestamp', y=['low', 'medium', 'high'], labels={'value':'confidence','variable':'class'}, template='plotly_dark')
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
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

    # update summary cards
    try:
        # today's attendance count
        if st.session_state.storage:
            from datetime import date
            today = date.today()
            atts = st.session_state.storage.get_attendance(limit=None)
            today_count = 0
            persons = set()
            for a in atts:
                try:
                    ts = pd.to_datetime(a.get('timestamp')).date()
                except Exception:
                    ts = None
                if ts == today:
                    today_count += 1
                    if a.get('person'):
                        persons.add(a.get('person'))
            placeholders['today_att_card'].markdown(f"<div class=\"card\"><div class=\"large-title\">Today's Attendance</div><div class=\"small-muted\">{today_count} present today</div></div>", unsafe_allow_html=True)
            # total events
            evs = st.session_state.storage.get_events(limit=None)
            placeholders['total_events_card'].markdown(f"<div class=\"card\"><div class=\"large-title\">Total Events</div><div class=\"small-muted\">{len(evs)}</div></div>", unsafe_allow_html=True)
        else:
            placeholders['today_att_card'].text("No storage")
            placeholders['total_events_card'].text("No storage")
    except Exception:
        placeholders['today_att_card'].text('N/A')
        placeholders['total_events_card'].text('N/A')

    # current occupancy card
    try:
        cur_label = st.session_state.label
        cur_conf = max(st.session_state.confidences.get('medium', 0.0), st.session_state.confidences.get('high', 0.0))
        # large gauge indicator (plotly)
        import plotly.graph_objects as go
        val = float(cur_conf) * 100.0
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=val,
            delta={'reference': 0},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#00E5FF"}},
            title={'text': f"{cur_label.upper()}"}
        ))
        gauge.update_layout(height=260, margin=dict(t=10,b=10,l=10,r=10), paper_bgcolor='rgba(0,0,0,0)', font={'color':'#e6eef3'})
        placeholders['current_occ_card'].plotly_chart(gauge, use_container_width=True)
    except Exception:
        placeholders['current_occ_card'].text('N/A')

    # History tab content: load from persistent storage
    try:
        if st.session_state.storage:
            oh = st.session_state.storage.get_occupancy_history(limit=2000)[::-1]
            if oh:
                dfh = pd.DataFrame([{'timestamp': r['timestamp'], 'low': r['low'], 'medium': r['medium'], 'high': r['high']} for r in oh])
                dfh['timestamp'] = pd.to_datetime(dfh['timestamp'])
                figh = px.line(dfh, x='timestamp', y=['low', 'medium', 'high'], labels={'value':'confidence','variable':'class'})
                placeholders['history_chart'].plotly_chart(figh, use_container_width=True)
            else:
                placeholders['history_chart'].text('No persisted history')

            persisted_events = st.session_state.storage.get_events()
            if persisted_events:
                # render recent events with thumbnails (most recent first)
                # thumbnail gallery + timeline
                thumbs = [e for e in persisted_events if e.get('thumbnail_path')]
                if thumbs:
                    st.markdown('<div class="thumb-grid">', unsafe_allow_html=True)
                    for e in thumbs[:40]:
                        try:
                            p = Path(e.get('thumbnail_path'))
                            if p.exists():
                                st.image(str(p), use_column_width=True)
                            else:
                                st.write('missing')
                        except Exception:
                            st.write('err')
                    st.markdown('</div>', unsafe_allow_html=True)

                # event timeline chart
                try:
                    import plotly.express as _px
                    tev = persisted_events[:200]
                    dftev = pd.DataFrame([{'timestamp': r['timestamp'], 'level': r['level'], 'msg': r['message']} for r in tev])
                    if not dftev.empty:
                        dftev['timestamp'] = pd.to_datetime(dftev['timestamp'])
                        tfig = _px.scatter(dftev, x='timestamp', y=[1]*len(dftev), color='level', hover_data=['msg'], height=200)
                        tfig.update_yaxes(visible=False)
                        tfig.update_layout(margin=dict(t=10,b=10,l=10,r=10), template='plotly_dark')
                        placeholders['history_events'].plotly_chart(tfig, use_container_width=True)
                except Exception:
                    pass
            else:
                placeholders['history_events'].text('No persisted events')

            # Attendance table
            att = st.session_state.storage.get_attendance(limit=1000)
            if att:
                df_att = pd.DataFrame(att)
                # person filter
                if 'person_filter' in locals() and person_filter:
                    df_att = df_att[df_att['person'].astype(str).str.contains(person_filter, case=False, na=False)]
                # date range filter
                if 'date_range' in locals() and date_range:
                    try:
                        if len(date_range) == 2:
                            start, end = date_range
                            df_att['timestamp'] = pd.to_datetime(df_att['timestamp'])
                            df_att = df_att[(df_att['timestamp'].dt.date >= start) & (df_att['timestamp'].dt.date <= end)]
                    except Exception:
                        pass
                # search
                if 'att_search' in locals() and att_search:
                    df_att = df_att[df_att.apply(lambda r: att_search.lower() in str(r.values).lower(), axis=1)]
                # render attendance rows with thumbnail preview
                for idx, r in df_att.head(200).iterrows():
                    c1, c2 = st.columns([1, 6])
                    tp = r.get('thumbnail_path')
                    if tp and Path(tp).exists():
                        c1.image(str(Path(tp)), use_column_width=True)
                    else:
                        c1.text('')
                    c2.write(r.to_dict())
            else:
                placeholders['attendance_table'].text('No attendance records')

            # Export button handling
            # Events export button
            export_events = st.button("Export events CSV")
            if export_events:
                try:
                    from pathlib import Path
                    out = Path.cwd() / 'data' / 'events_export.csv'
                    st.session_state.storage.export_events_csv(out)
                    placeholders['history_events'].markdown(f"Exported events to {out}")
                except Exception as e:
                    placeholders['history_events'].text(f"Export failed: {e}")

            # Attendance export handling
            if placeholders['attendance_export_button']:
                try:
                    from pathlib import Path
                    out = Path.cwd() / 'data' / 'attendance_export.csv'
                    st.session_state.storage.export_attendance_csv(out)
                    placeholders['attendance_export_placeholder'].markdown(f"Exported to {out}")
                except Exception as e:
                    placeholders['attendance_export_placeholder'].text(f"Export failed: {e}")
    except Exception:
        placeholders['history_chart'].text('Failed to load persisted history')
        placeholders['history_events'].text('Failed to load persisted events')


if __name__ == '__main__':
    main()
