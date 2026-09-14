"""Read-only Jetson input-voltage monitoring; battery charge is NOT inferred.

INA3221 POM_5V_IN is a regulated supply. A USB power bank may hold it near 5 V
until abrupt shutdown. Actual battery state requires a separate fuel gauge.
"""
import math
import time


class InputVoltage:
    def __init__(self):
        import os
        self.fd = None
        device='/sys/bus/i2c/devices/6-0040'
        with open(device+'/name') as source:
            if source.read().strip()!='ina3221x':
                raise RuntimeError('Expected Jetson INA3221 input monitor')
        with open(device+'/of_node/channel@0/ti,rail-name','rb') as source:
            if source.read().rstrip(b'\0') not in (b'POM_5V_IN',b'VDD_IN'):
                raise RuntimeError('Channel 1 is not the confirmed input rail')
        self.fd=os.open('/dev/i2c-6',os.O_RDWR)
        try:
            if self.read(0xfe)!=0x5449 or self.read(0xff)!=0x3220:
                raise RuntimeError('INA3221 identity mismatch')
        except BaseException:
            self.close()
            raise

    def read(self,register):
        import ctypes
        import fcntl
        from hardware import Message,Transfer
        address=ctypes.create_string_buffer(bytes([register]),1)
        output=ctypes.create_string_buffer(2)
        messages=(Message*2)(Message(0x40,0,1,ctypes.addressof(address)),
                             Message(0x40,1,2,ctypes.addressof(output)))
        fcntl.ioctl(self.fd,0x0707,Transfer(messages,2))
        return int.from_bytes(output.raw,'big')

    def sample(self):
        config=self.read(0)
        if not config & 0x4000 or config & 7 not in (6,7):
            raise RuntimeError('Input monitor is not continuously converting bus voltage')
        periods=(.00014,.000204,.000332,.000588,.0011,.002116,.004156,.008244)
        averages=(1,4,16,64,128,256,512,1024)
        channels=sum(bool(config & (1<<bit)) for bit in (12,13,14))
        period=channels*averages[(config>>9)&7]*(periods[(config>>6)&7]+(periods[(config>>3)&7] if config&7==7 else 0))
        if period>.2:
            raise RuntimeError('Input monitor conversion period exceeds 200 ms')
        raw=self.read(2)
        if self.read(0)!=config:
            raise RuntimeError('Power-monitor configuration changed during read')
        if raw & 0x8000:
            raise RuntimeError('Negative input-voltage reading')
        return dict(time=time.monotonic(),voltage_v=(raw>>3)*.008,
                    source='INA3221 POM_5V_IN',conversion_period_seconds=period,
                    battery_percent=None,battery_state='unknown_no_fuel_gauge')

    def close(self):
        import os
        if self.fd is not None:
            os.close(self.fd)
            self.fd=None


class PowerGuard:
    """Conservative experiment policy, not a battery discharge curve."""
    def __init__(self):
        self.reason=None
        self.samples=0
        self.minimum=None

    def update(self,sample,now):
        voltage=sample.get('voltage_v')
        timestamp=sample.get('time')
        valid=(isinstance(voltage,(int,float)) and not isinstance(voltage,bool)
               and math.isfinite(voltage) and isinstance(timestamp,(int,float))
               and math.isfinite(timestamp) and 0<=now-timestamp<=.6)
        if not valid or sample.get('fault'):
            self.reason=self.reason or 'power_telemetry_unavailable'
        else:
            self.samples+=1
            self.minimum=voltage if self.minimum is None else min(self.minimum,voltage)
            if voltage<4.8:
                self.reason=self.reason or 'input_undervoltage'
            elif voltage>5.5:
                self.reason=self.reason or 'input_overvoltage'
        return dict(sample, motion_allowed=self.reason is None and self.samples>=3,
                    stop_latched=self.reason is not None,stop_reason=self.reason,
                    warning=bool(valid and voltage<4.9),minimum_voltage_v=self.minimum,
                    warn_below_v=4.9,stop_below_v=4.8,stale_after_seconds=.6)


def power_permitted(values,now):
    timestamp,voltage,permitted=values
    return (permitted==1 and math.isfinite(timestamp) and 0<=now-timestamp<=.6
            and math.isfinite(voltage) and 4.8<=voltage<=5.5)


def read_shared(shared,previous):
    # A killed sensor process must not leave the motor watchdog blocked on its
    # shared lock. Reuse the last sample briefly; its .6-second expiry still wins.
    lock=shared.get_lock()
    if not lock.acquire(True,.001):
        return previous
    try:return list(shared[:])
    finally:lock.release()


def worker(shared,q,stop,log_path):
    import json
    import queue
    sensor=None
    guard=PowerGuard()
    try:
        sensor=InputVoltage()
        with open(log_path,'a',buffering=1) as log:
            while not stop.is_set():
                try:
                    sample=sensor.sample()
                except Exception as exc:
                    sample=dict(time=time.monotonic(),fault=str(exc),battery_percent=None)
                status=guard.update(sample,time.monotonic())
                with shared.get_lock():
                    shared[:]=[status['time'],status.get('voltage_v',0.),float(status['motion_allowed'])]
                log.write(json.dumps(status)+'\n')
                try:q.put_nowait(status)
                except queue.Full:pass
                stop.wait(.2)
    except Exception as exc:
        with shared.get_lock():shared[:]=[time.monotonic(),0.,0.]
        try:q.put_nowait(dict(time=time.monotonic(),fault=str(exc),motion_allowed=False,stop_latched=True,battery_percent=None))
        except queue.Full:pass
    finally:
        if sensor is not None:sensor.close()
        with shared.get_lock():shared[2]=0.


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=10.)
    parser.add_argument('--output')
    args=parser.parse_args()
    if not 0<args.seconds<=3600:parser.error('seconds must be 0 to 3600')
    sensor=InputVoltage();guard=PowerGuard();started=time.monotonic()
    log=open(args.output,'w') if args.output else None
    try:
        while time.monotonic()-started<args.seconds:
            sample=guard.update(sensor.sample(),time.monotonic())
            line=json.dumps(sample)
            if log:log.write(line+'\n');log.flush()
            print(line,flush=True)
            time.sleep(.2)
    finally:
        sensor.close()
        if log:log.close()
