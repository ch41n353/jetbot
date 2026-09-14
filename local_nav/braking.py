"""Experimental one-shot braking and post-stop arrival assessment."""
import math


def braking_distance(speed_cm_s, frame_age, frame_interval):
    values = (speed_cm_s, frame_age, frame_interval)
    if not all(math.isfinite(v) for v in values):
        raise RuntimeError('Nonfinite braking state')
    if not 0 <= frame_age <= .18 or not 0 < frame_interval <= .18:
        raise RuntimeError('Stale observation for braking')
    # Observation age plus half an update and an initial 40 ms coast estimate.
    # This is an experimental feed-forward estimate, not a calibrated brake model.
    horizon = min(.18, frame_age + .5 * frame_interval + .04)
    return min(2., max(0., speed_cm_s) * horizon)


class SettlingCheck:
    def __init__(self, target):
        self.target = target
        self.samples = []

    def update(self, timestamp, x, z):
        if not all(math.isfinite(v) for v in (timestamp, x, z)):
            raise RuntimeError('Nonfinite settling state')
        if self.samples and timestamp <= self.samples[-1][0]:
            raise RuntimeError('Settling requires distinct increasing frames')
        self.samples.append((timestamp, x, z))
        self.samples = [s for s in self.samples if timestamp - s[0] <= .3]
        if len(self.samples) < 3 or timestamp - self.samples[0][0] < .18:
            return None
        xs, zs = [s[1] for s in self.samples], [s[2] for s in self.samples]
        if max(xs) - min(xs) > .10 or max(zs) - min(zs) > .10:
            return None
        error = z - self.target
        return dict(outcome='goal_reached' if abs(error) <= 1 else
                    ('stopped_short' if error < 0 else 'overshot_goal'),
                    final_position_cm=[x, z], final_error_cm=error,
                    settled_window_seconds=timestamp - self.samples[0][0])
