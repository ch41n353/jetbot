#!/usr/bin/env python3
"""Measure how fast the pack is charging, and how long it has left.

Reads the same INA219 pack sensor the power guard uses, through the service's
status socket, so this needs no extra hardware and works while the robot is
idle.

Two things to keep in mind when reading the output.

The Jetson is powered from the same pack, so what this measures is the *net*
rate: charger output minus whatever the board is drawing. That is the number
that matters for "when can I drive again", but it is not the charger's own
output, and a charger that looks slow here may simply be losing most of its
current to the platform.

And voltage is a soft proxy for state of charge while current is flowing. A
pack under charge reads high immediately and sags back when the charger comes
off, so the endpoint worth waiting for is the *plateau* -- the point where the
rise flattens because the charger has moved from constant current to constant
voltage -- rather than any particular reading.

Artifacts go to the USB run root, never the SD card. Flushed, not fsync'd.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, 'local_nav'))

# A 3S lithium pack: 4.2 V per cell fully charged, 3.0 V empty. The charger
# holds the top figure while it finishes, so treat it as the target, not a
# reading to expect to see settle at once the charger is removed.
CELLS = 3
FULL_V = 4.2 * CELLS
NOMINAL_V = 3.7 * CELLS


def pack_voltage():
    import fetch
    return float(fetch.call('status')['power']['pack_voltage_v'])


def fit(samples):
    """Least-squares volts per hour over (seconds, volts) samples."""
    if len(samples) < 3:
        return None
    n = float(len(samples))
    mean_t = sum(t for t, _ in samples) / n
    mean_v = sum(v for _, v in samples) / n
    spread = sum((t - mean_t) ** 2 for t, _ in samples)
    if spread <= 0.:
        return None
    slope = sum((t - mean_t) * (v - mean_v) for t, v in samples) / spread
    return slope * 3600.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=float, default=30.,
                        help='seconds between samples')
    parser.add_argument('--report', type=float, default=300.,
                        help='seconds between printed summaries')
    parser.add_argument('--window', type=float, default=900.,
                        help='seconds of history the rate is fitted over')
    parser.add_argument('--target', type=float, default=FULL_V)
    parser.add_argument('--hours', type=float, default=6.)
    parser.add_argument('--label', default='',
                        help='tag every record, so two chargers can be compared '
                             'from one file without splitting it by timestamp')
    args = parser.parse_args()

    if not os.path.ismount('/mnt/robotlogs'):
        raise SystemExit('robot log volume not mounted')
    with open('/mnt/robotlogs/current-search.json') as handle:
        root = json.load(handle)['root']
    record = open(os.path.join(root, 'charge-watch.jsonl'), 'a')

    started = time.time()
    samples = []
    first = None
    last_report = 0.
    print('watching pack charge | sample %.0fs | report %.0fs | target %.2f V%s'
          % (args.interval, args.report, args.target,
             ' | label %s' % args.label if args.label else ''))
    while time.time() - started < args.hours * 3600.:
        try:
            volts = pack_voltage()
        except Exception as exc:
            print('sample failed: %s' % exc)
            time.sleep(args.interval)
            continue
        now = time.time() - started
        if first is None:
            first = volts
        samples.append((now, volts))
        samples[:] = [s for s in samples if now - s[0] <= args.window]
        record.write(json.dumps(dict(elapsed_s=round(now, 1), pack_v=volts,
                                     label=args.label, at=time.time())) + '\n')
        record.flush()

        if now - last_report >= args.report:
            last_report = now
            rate = fit(samples)
            line = ('t+%5.1f min  pack %.3f V (%.2f/cell)  net %+.3f V total'
                    % (now / 60., volts, volts / CELLS, volts - first))
            if rate is not None:
                line += '  rate %+.3f V/h' % rate
                if rate > .02 and volts < args.target:
                    line += '  -> %.2f V in %.1f h' % (
                        args.target, (args.target - volts) / rate)
                elif abs(rate) <= .02:
                    # Flat is the interesting state: either finished, or the
                    # charger is only keeping up with the Jetson's draw.
                    line += '  (flat: charged, or charger only offsetting load)'
                elif rate < 0:
                    line += '  (FALLING: draw exceeds charger output)'
            print(line)
            sys.stdout.flush()
        time.sleep(args.interval)

    record.close()


if __name__ == '__main__':
    main()
