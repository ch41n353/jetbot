import ctypes
import fcntl
import importlib.util
import os
import struct
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def robot():
    sys.path.append('/home/jetbot/jetbot-env/lib/python3.6/site-packages')
    for name in ('motor', 'robot'):
        full = 'jetbot.'+name
        spec = importlib.util.spec_from_file_location(full, os.path.join(ROOT, 'jetbot', name+'.py'))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return mod.Robot()


class Message(ctypes.Structure):
    _fields_ = [('addr', ctypes.c_uint16), ('flags', ctypes.c_uint16),
                ('length', ctypes.c_uint16), ('buffer', ctypes.c_void_p)]


class Transfer(ctypes.Structure):
    _fields_ = [('messages', ctypes.POINTER(Message)), ('count', ctypes.c_uint32)]


class BNO055:
    def __init__(self):
        self.fd = os.open('/dev/i2c-0', os.O_RDWR)
        self.original = None
        try:
            if self.read(0, 1)[0] != 0xa0:
                raise RuntimeError('BNO055 not found at bus 0 address 0x28')
            self.original = self.read(0x3d, 1)[0] & 15
            self.write(0x3d, 0)
            time.sleep(.03)
            self.units = self.read(0x3b, 1)[0]
            self.write(0x3d, 12)
            time.sleep(.7)
        except BaseException:
            self.close()
            raise

    def read(self, reg, size):
        address = ctypes.create_string_buffer(bytes([reg]), 1)
        output = ctypes.create_string_buffer(size)
        messages = (Message*2)(Message(0x28, 0, 1, ctypes.addressof(address)),
                               Message(0x28, 1, size, ctypes.addressof(output)))
        fcntl.ioctl(self.fd, 0x0707, Transfer(messages, 2))
        return output.raw

    def write(self, reg, value):
        buf = ctypes.create_string_buffer(bytes([reg, value]), 2)
        messages = (Message*1)(Message(0x28, 0, 2, ctypes.addressof(buf)))
        fcntl.ioctl(self.fd, 0x0707, Transfer(messages, 1))

    def sample(self):
        data = self.read(8, 24)
        def vector(offset, divisor):
            return [v/divisor for v in struct.unpack_from('<hhh', data, offset)]
        status = self.read(0x35, 6)
        cal = status[0]
        return dict(acceleration=vector(0, 1000/9.80665 if self.units & 1 else 100),
                    gyro=vector(12, 900 if self.units & 2 else 16),
                    euler=vector(18, 900 if self.units & 4 else 16),
                    gyro_units='rad/s' if self.units & 2 else 'deg/s',
                    euler_units='rad' if self.units & 4 else 'deg',
                    calibration=[(cal >> shift) & 3 for shift in (6,4,2,0)],
                    self_test=status[1], system_status=status[4], error=status[5])

    def close(self):
        try:
            if self.original is not None:
                self.write(0x3d, 0)
                time.sleep(.03)
                self.write(0x3d, self.original)
                time.sleep(.03)
        finally:
            os.close(self.fd)
