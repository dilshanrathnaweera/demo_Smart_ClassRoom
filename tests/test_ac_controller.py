from smart_classroom.ac_controller import ACController


def test_ac_controller_transitions():
    ctrl = ACController()
    assert ctrl.current_status().on is False

    evt = ctrl.update('low', {'low': 0.9, 'medium': 0.05, 'high': 0.05})
    assert ctrl.current_status().on is False
    assert ctrl.current_status().temperature is None

    evt = ctrl.update('medium', {'low': 0.1, 'medium': 0.8, 'high': 0.1})
    assert ctrl.current_status().on is True
    assert ctrl.current_status().temperature == 24.0

    evt = ctrl.update('high', {'low': 0.05, 'medium': 0.15, 'high': 0.8})
    assert ctrl.current_status().on is True
    assert ctrl.current_status().temperature == 20.0
