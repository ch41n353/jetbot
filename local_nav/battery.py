"""Read-only battery-pack and Jetson input-voltage monitoring.

INA3221 POM_5V_IN is a regulated supply. A USB power bank may hold it near 5 V
until abrupt shutdown. Actual battery state requires a separate fuel gauge.
"""
import math
import os
import time


class InputVoltage:
    def __init__(self):
        import os
        self.fd = None
        self.address=0x40
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
        messages=(Message*2)(Message(self.address,0,1,ctypes.addressof(address)),
                             Message(self.address,1,2,ctypes.addressof(output)))
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


class PackVoltage(InputVoltage):
    """Read the discovered INA219-compatible monitor on the motor-board bus.

    The 0x41 address, default0x399f configuration and12.276V reading match the
    Waveshare three-cell JetBot board. No accurate SOC/current is inferred.
    """
    def __init__(self):
        import os
        self.address=0x41
        self.fd=os.open('/dev/i2c-1',os.O_RDWR)
        try:self.sample()
        except BaseException:self.close();raise

    def sample(self):
        config=self.read(0)
        if config & 0xc000 or config & 7 not in (6,7):
            raise RuntimeError('Pack monitor is not a continuously converting INA219 configuration')
        def duration(code):
            if code<4:return (.000084,.000148,.000276,.000532)[code]
            if code>=8:return .000532*(2**(code-8))
            raise RuntimeError('Unsupported pack ADC configuration')
        period=duration((config>>7)&15)+(duration((config>>3)&15) if config&7==7 else 0)
        if period>.2:raise RuntimeError('Pack conversion period exceeds 200 ms')
        raw=self.read(2)
        if self.read(0)!=config:raise RuntimeError('Pack monitor configuration changed during read')
        if not raw & 2:raise RuntimeError('Pack voltage conversion is not ready')
        return dict(pack_time=time.monotonic(),pack_voltage_v=(raw>>3)*.004,
                    pack_source='INA219-compatible bus1 address0x41',
                    pack_conversion_period_seconds=period,pack_required=True,
                    battery_percent=None,battery_state='voltage_only_no_soc_estimate',
                    pack_profile='Waveshare-compatible 3S inferred from hardware; board model unconfirmed')


class PowerSensors:
    def __init__(self):
        self.supply=InputVoltage()
        try:self.pack=PackVoltage()
        except BaseException:self.supply.close();raise

    def sample(self):
        sample=self.supply.sample()
        sample['input_time']=sample['time']
        sample.update(self.pack.sample())
        sample['time']=min(sample['time'],sample['pack_time'])
        return sample

    def close(self):
        self.supply.close();self.pack.close()


class PowerGuard:
    """Conservative experiment policy, not a battery discharge curve."""
    def __init__(self,require_pack=False):
        self.reason=None
        self.samples=0
        self.minimum=None
        self.pack_minimum=None
        self.require_pack=require_pack

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
            elif voltage>5.25:
                self.reason=self.reason or 'input_overvoltage'
        pack=sample.get('pack_voltage_v')
        pack_valid=(isinstance(pack,(int,float)) and not isinstance(pack,bool) and math.isfinite(pack))
        if self.require_pack or sample.get('pack_required'):
            if not pack_valid:
                self.reason=self.reason or 'pack_telemetry_unavailable'
            else:
                self.pack_minimum=pack if self.pack_minimum is None else min(self.pack_minimum,pack)
                if pack<10.8:self.reason=self.reason or 'battery_pack_low'
                elif pack>12.9:self.reason=self.reason or 'battery_pack_above_profile'
        return dict(sample, motion_allowed=self.reason is None and self.samples>=3,
                    stop_latched=self.reason is not None,stop_reason=self.reason,
                    warning=bool((valid and voltage<4.9) or (pack_valid and pack<11.4)),
                    minimum_voltage_v=self.minimum,minimum_pack_voltage_v=self.pack_minimum,
                    warn_below_v=4.9,stop_below_v=4.8,stop_above_v=5.25,pack_warn_below_v=11.4,
                    pack_stop_below_v=10.8,pack_stop_above_v=12.9,stale_after_seconds=.6,
                    power_policy_revision=2)


def power_permitted(values,now):
    timestamp,voltage,permitted=values
    return (permitted==1 and math.isfinite(timestamp) and 0<=now-timestamp<=.6
            and math.isfinite(voltage) and 4.8<=voltage<=5.25)


def read_shared(shared,previous):
    # A killed sensor process must not leave the motor watchdog blocked on its
    # shared lock. Reuse the last sample briefly; its .6-second expiry still wins.
    lock=shared.get_lock()
    if not lock.acquire(True,.001):
        return previous
    try:return list(shared[:])
    finally:lock.release()


LOG_MAX_BYTES = 16 * 1024 * 1024   # roll at 16 MB, about an hour of samples
LOG_KEEP = 3                       # this file plus three older ones: ~64 MB total


def _roll(path, keep=LOG_KEEP):
    """Shuffle power.jsonl -> .1 -> .2 -> ... and drop the oldest.

    The worker appends a record five times a second, every field of every
    sample including the static thresholds and the profile string. That is
    roughly 345 MB a day, and it had reached 287 MB before anyone looked. The
    recent history is what anyone actually reads back, so keep a bounded window
    of it rather than everything since boot.
    """
    try:
        oldest = '%s.%d' % (path, keep)
        if os.path.exists(oldest):
            os.unlink(oldest)
        for index in range(keep - 1, 0, -1):
            older = '%s.%d' % (path, index)
            if os.path.exists(older):
                os.rename(older, '%s.%d' % (path, index + 1))
        if os.path.exists(path):
            os.rename(path, path + '.1')
        return True
    except OSError:
        # Rotation is housekeeping. The power monitor is not: if it dies the
        # service loses its guard and motion latches off, so a failure here
        # must leave the worker writing to the handle it already has.
        return False


def worker(shared,q,stop,log_path):
    import json
    import queue
    sensor=None
    guard=PowerGuard(require_pack=True)
    try:
        sensor=PowerSensors()
        log = open(log_path,'a',buffering=1)
        try:
            written = os.path.getsize(log_path)
        except OSError:
            written = 0
        try:
            while not stop.is_set():
                try:
                    sample=sensor.sample()
                except Exception as exc:
                    sample=dict(time=time.monotonic(),fault=str(exc),battery_percent=None)
                status=guard.update(sample,time.monotonic())
                with shared.get_lock():
                    shared[:]=[status['time'],status.get('voltage_v',0.),float(status['motion_allowed'])]
                line=json.dumps(status)+'\n'
                log.write(line)
                written+=len(line)
                if written>=LOG_MAX_BYTES:
                    log.close()
                    if _roll(log_path):
                        written=0
                    log=open(log_path,'a',buffering=1)
                    if written:
                        try:written=os.path.getsize(log_path)
                        except OSError:written=0
                try:q.put_nowait(status)
                except queue.Full:pass
                stop.wait(.2)
        finally:
            try:log.close()
            except Exception:pass
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
    sensor=PowerSensors();guard=PowerGuard(require_pack=True);started=time.monotonic()
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
