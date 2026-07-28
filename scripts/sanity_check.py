from smart_classroom.ac_controller import ACController


def run():
    ctrl = ACController()
    print('Initial status:', ctrl.current_status())
    evt = ctrl.update('medium', {'low':0.1,'medium':0.85,'high':0.05})
    print('After medium:', ctrl.current_status())
    print('Event:', evt)


if __name__ == '__main__':
    run()
