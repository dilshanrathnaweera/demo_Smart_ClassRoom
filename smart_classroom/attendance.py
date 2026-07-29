from collections import deque
from datetime import datetime, timedelta
from typing import Deque, Dict, Optional


class AttendanceController:
    """Detects presence based on occupancy confidences.

    Rules:
    - A 'present' detection fires when medium/high confidence is >= `confidence_threshold`
      continuously for at least `required_duration` seconds.
    - After a detection, the controller remains in the `present` state until the
      confidences drop below the threshold for `absence_duration` seconds.
    - When a detection occurs, a record dict is returned with keys: person, timestamp, level, confidence
    """

    def __init__(self, required_duration_seconds: int = 8, confidence_threshold: float = 0.5, absence_duration_seconds: int = 3, person: str = "unknown"):
        self.required_duration = timedelta(seconds=required_duration_seconds)
        self.confidence_threshold = confidence_threshold
        self.absence_duration = timedelta(seconds=absence_duration_seconds)
        self.person = person

        # store recent samples as (timestamp: datetime, presence_confidence: float, level:str)
        self._samples: Deque = deque()
        self._present = False
        self._last_present_ts: Optional[datetime] = None

    def _presence_conf(self, confidences: Dict[str, float]) -> float:
        return max(confidences.get('medium', 0.0), confidences.get('high', 0.0))

    def _dominant_level(self, confidences: Dict[str, float]) -> str:
        # pick the label with highest confidence
        items = [('low', confidences.get('low', 0.0)), ('medium', confidences.get('medium', 0.0)), ('high', confidences.get('high', 0.0))]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[0][0]

    def process_sample(self, confidences: Dict[str, float], timestamp: Optional[datetime] = None) -> Optional[Dict]:
        """Process a new occupancy sample. If attendance is detected, return a record dict.

        The returned dict has keys: `person`, `timestamp`, `level`, `confidence`.
        """
        ts = timestamp or datetime.utcnow()
        pconf = self._presence_conf(confidences)
        level = self._dominant_level(confidences)

        # append and prune old samples
        self._samples.append((ts, pconf, level))
        # keep enough history to evaluate both presence and absence windows
        cutoff = ts - (self.required_duration + self.absence_duration)
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        # determine continuous presence duration where pconf >= threshold
        qualifying = [s for s in self._samples if s[1] >= self.confidence_threshold]
        if qualifying:
            duration = qualifying[-1][0] - qualifying[0][0]
        else:
            duration = timedelta(seconds=0)

        # if not present and we have enough continuous qualifying duration -> mark present
        if not self._present and duration >= self.required_duration:
            self._present = True
            self._last_present_ts = ts
            # choose confidence and level from latest qualifying sample
            last_q = qualifying[-1]
            rec = {'person': self.person, 'timestamp': ts, 'level': last_q[2], 'confidence': last_q[1]}
            return rec

        # if present, check for absence condition to reset state
        if self._present:
            # absence samples: those below threshold
            absence_samples = [s for s in self._samples if s[1] < self.confidence_threshold]
            if absence_samples:
                absence_duration = absence_samples[-1][0] - absence_samples[0][0]
            else:
                absence_duration = timedelta(seconds=0)

            if absence_duration >= self.absence_duration:
                self._present = False

        return None


__all__ = ["AttendanceController"]
