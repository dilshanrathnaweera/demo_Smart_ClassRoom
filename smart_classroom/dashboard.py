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

from smart_classroom.inference import OnnxClassifier
from smart_classroom.ac_controller import ACController
from smart_classroom.attendance import AttendanceController
from smart_classroom.thumbnail import save_thumbnail


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


def render_dashboard():
    # top summary cards
    c1, c2, c3 = st.columns(3)
    with c1:
        today_att_card = st.empty()
    with c2:
        total_events_card = st.empty()
    with c3:
        current_occ_card = st.empty()

    # layout with tabs: Live / History / Attendance
    tab_live, tab_history, tab_att = st.tabs(["Live", "History", "Attendance"])

    with tab_live:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.header("Live Camera Feed")
            img_placeholder = st.empty()
            st.markdown("---")
            st.header("Occupancy History (recent)")
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

    with tab_history:
        st.header("Full Occupancy History")
        history_chart = st.empty()
        st.markdown("---")
        st.header("Event Log (persisted)")
        history_events = st.empty()

    with tab_att:
        st.header("Attendance Records")
        # filters: date range and person
        fcol1, fcol2, fcol3 = st.columns([1, 2, 1])
        with fcol1:
            date_range = st.date_input("Date range", [])
        with fcol2:
            person_filter = st.text_input("Person (leave empty for all)")
        with fcol3:
            att_search = st.text_input("Search attendance (level/confidence)")

        st.markdown("---")
        att_table = st.empty()
        st.markdown("---")
        st.header("Export")
        export_att = st.button("Export attendance CSV")
        att_export_placeholder = st.empty()

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

    # history chart is also available in the History tab; here show a small recent chart
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
            placeholders['today_att_card'].metric("Today's Attendance", today_count)
            # total events
            evs = st.session_state.storage.get_events(limit=None)
            placeholders['total_events_card'].metric('Total Events', len(evs))
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
        placeholders['current_occ_card'].metric('Current Occupancy', cur_label, f"{cur_conf:.2f}")
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
                for e in persisted_events[:50]:
                    cols = st.columns([1, 5])
                    thumb = e.get('thumbnail_path')
                    if thumb:
                        try:
                            p = Path(thumb)
                            if p.exists():
                                cols[0].image(str(p), use_column_width=True)
                            else:
                                cols[0].text('missing')
                        except Exception:
                            cols[0].text('err')
                    else:
                        cols[0].text('')
                    cols[1].markdown(f"**{e.get('timestamp')}** — **{e.get('level')}**  \n{e.get('message')}  \nlow:{e.get('confidences',{}).get('low',0):.2f} medium:{e.get('confidences',{}).get('medium',0):.2f} high:{e.get('confidences',{}).get('high',0):.2f}")
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
