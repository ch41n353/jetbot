"""Hardware-independent motor lease validation."""
import math


def allowed(command, now, enabled):
    if not enabled:
        return False
    try:
        values = [command[k] for k in ('left', 'right', 'issued', 'expires', 'camera_time', 'imu_time')]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            return False
        return (abs(command['left']) <= .3 and abs(command['right']) <= .3
                and 0 <= now-command['issued'] <= .25
                and now < command['expires'] <= command['issued']+.25
                and 0 <= now-command['camera_time'] <= .5
                and 0 <= now-command['imu_time'] <= .2)
    except (KeyError, TypeError):
        return False
