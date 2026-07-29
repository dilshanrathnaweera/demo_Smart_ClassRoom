from datetime import datetime, timedelta
from smart_classroom.attendance import AttendanceController

def make_conf(medium=0.0, high=0.0):
    return {'low': 1.0 - (medium + high), 'medium': medium, 'high': high}

ctrl = AttendanceController(required_duration_seconds=1, confidence_threshold=0.5, absence_duration_seconds=1, person='student1')
t0 = datetime.utcnow()
print('t0', t0)
print('sample0', ctrl.process_sample(make_conf(medium=0.3), t0))
print('sample1 at +1', ctrl.process_sample(make_conf(medium=0.6), t0 + timedelta(seconds=1)))
print('sample2 at +2.1', ctrl.process_sample(make_conf(medium=0.7), t0 + timedelta(seconds=2.1)))
