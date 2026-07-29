from datetime import datetime, timedelta
from smart_classroom.attendance import AttendanceController


def make_conf(medium=0.0, high=0.0):
    return {'low': 1.0 - (medium + high), 'medium': medium, 'high': high}


def test_attendance_detects_persistent_presence():
    ctrl = AttendanceController(required_duration_seconds=2, confidence_threshold=0.5, absence_duration_seconds=1, person='student1')
    t0 = datetime.utcnow()
    # feed samples below threshold
    assert ctrl.process_sample(make_conf(medium=0.3), t0) is None
    # feed samples above threshold for > required_duration
    rec = ctrl.process_sample(make_conf(medium=0.6), t0 + timedelta(seconds=1))
    assert rec is None
    rec = ctrl.process_sample(make_conf(medium=0.7), t0 + timedelta(seconds=2.1))
    assert rec is not None
    assert rec['person'] == 'student1'
    assert rec['level'] in ('medium','high')


def test_attendance_resets_on_absence():
    ctrl = AttendanceController(required_duration_seconds=1, confidence_threshold=0.5, absence_duration_seconds=1, person='student2')
    t0 = datetime.utcnow()
    # get present
    ctrl.process_sample(make_conf(medium=0.6), t0)
    ctrl.process_sample(make_conf(medium=0.7), t0 + timedelta(seconds=1.2))
    # now feed absence samples
    ctrl.process_sample(make_conf(medium=0.1), t0 + timedelta(seconds=2))
    ctrl.process_sample(make_conf(medium=0.1), t0 + timedelta(seconds=3.5))
    # after absence duration, feeding presence again should re-detect
    rec = ctrl.process_sample(make_conf(medium=0.8), t0 + timedelta(seconds=5))
    assert rec is None
    rec = ctrl.process_sample(make_conf(medium=0.9), t0 + timedelta(seconds=6.2))
    assert rec is not None
*** End Patch