import time
import tempfile
from pathlib import Path
import streamlit as st

from smart_classroom.storage import Storage
import importlib


def test_dashboard_storage_writer(tmp_path):
    # clear any previous session keys
    for k in list(st.session_state.keys()):
        del st.session_state[k]

    # prepare a storage db in temp
    db_file = tmp_path / 'db.sqlite'
    storage = Storage(db_file)

    # inject into session_state before init
    st.session_state.storage = storage

    # import dashboard module and call init_state
    from smart_classroom import dashboard
    importlib.reload(dashboard)
    dashboard.init_state()

    # ensure write queue exists
    assert 'storage_write_queue' in st.session_state

    # enqueue an event
    q = st.session_state.storage_write_queue
    q.put(('add_event', ('medium', 'test_event', {'low':0.1,'medium':0.8,'high':0.1}, None)))

    # wait for background writer to process
    time.sleep(1)

    events = storage.get_events()
    assert any(e['message'] == 'test_event' for e in events)

    storage.close()
