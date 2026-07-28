from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional


@dataclass
class ACStatus:
    on: bool
    temperature: Optional[float]


@dataclass
class ACEvent:
    timestamp: datetime
    level: str
    message: str
    confidences: Dict[str, float]


class ACController:
    """Simulates an AC system based on occupancy level.

    Rules:
      - low -> OFF
      - medium -> ON, 24°C
      - high -> ON, 20°C
    """

    def __init__(self):
        self.status = ACStatus(on=False, temperature=None)
        self.events: List[ACEvent] = []

    def update(self, level: str, confidences: Dict[str, float]):
        prev = (self.status.on, self.status.temperature)
        if level == "low":
            self.status.on = False
            self.status.temperature = None
            message = "AC turned OFF (low occupancy)"
        elif level == "medium":
            self.status.on = True
            self.status.temperature = 24.0
            message = "AC ON, set to 24°C (medium occupancy)"
        elif level == "high":
            self.status.on = True
            self.status.temperature = 20.0
            message = "AC ON, set to 20°C (high occupancy)"
        else:
            message = f"Unknown level: {level}"

        # log event only if changed or always log for traceability
        event = ACEvent(timestamp=datetime.utcnow(), level=level, message=message, confidences=confidences)
        self.events.append(event)
        return event

    def get_events(self) -> List[ACEvent]:
        return list(self.events)

    def current_status(self) -> ACStatus:
        return self.status


if __name__ == "__main__":
    # basic test
    ctrl = ACController()
    print(ctrl.current_status())
    ctrl.update('medium', {'low':0.1,'medium':0.8,'high':0.1})
    print(ctrl.current_status())
    print(ctrl.get_events())
