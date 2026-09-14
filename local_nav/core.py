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


class ControlGeneration:
    """Invalidate cooperating controllers after stop, errors or legacy commands."""
    def __init__(self):
        import uuid
        self.session_id = uuid.uuid4().hex
        self.epoch = 0

    def token(self):
        return dict(session_id=self.session_id, control_epoch=self.epoch)

    def invalidate(self):
        self.epoch += 1

    def validate(self, request):
        if any(key in request for key in self.token()):
            if any(request.get(key) != value for key, value in self.token().items()):
                raise ValueError('Control cancelled or service restarted; replan')
        else:
            self.invalidate()
