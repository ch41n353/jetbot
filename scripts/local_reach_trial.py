#!/usr/bin/env python3
"""Randomised reachability trial for the local trajectory executor.

The claim under test: within a 1 m radius the local planner should be able to
reach anywhere without failing. This samples SE(2) goals uniformly over that
disc, hands each one to `POST /api/local-execution` as a metric trajectory, and
records which shapes of request the executor refuses, stalls on, or completes.

The executor is not a collision checker -- it drives what it is given. Nothing
here knows about obstacles, so the robot must be standing in cleared carpet of
at least `radius` plus a chassis margin before this is run live. `--dry-run`
generates and validates the same trajectories with no motion at all.

Artifacts go to the USB run root named by /mnt/robotlogs/current-search.json,
never to the SD card. Records are flushed, not fsync'd.
"""
import argparse
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, 'local_nav'))

PLANNER = 'http://127.0.0.1:8770'
TERMINAL = ('completed', 'paused', 'failed', 'cancelled')

# Shapes worth separating in the report: a straight reach, a reach that has to
# turn first, a wandering polyline, and a rotation with no translation at all.
SHAPES = ('direct', 'turn_then_go', 'polyline', 'pure_turn')


def post(path, body, timeout=30):
    request = urllib.request.Request(
        PLANNER + path, data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def get(path, timeout=15):
    with urllib.request.urlopen(PLANNER + path, timeout=timeout) as response:
        return json.load(response)


def sample_point(rng, radius):
    """A point uniform over the disc of `radius`, as [right_cm, forward_cm]."""
    r = radius * math.sqrt(rng.random())
    a = rng.uniform(-math.pi, math.pi)
    return [r * math.sin(a), r * math.cos(a)]


def bearing_to(point):
    return math.degrees(math.atan2(point[0], point[1]))


def sample_trajectory(rng, radius, shape=None, max_path_cm=150.):
    """One random trajectory, every waypoint inside the disc of `radius`.

    `max_path_cm` bounds the driven path, not the reach. The executor gives an
    instruction 90 seconds total, so a wandering polyline across a 2 m-diameter
    disc runs out of clock for reasons that have nothing to do with whether the
    goal was reachable. Capping the path keeps a timeout meaning what it says.
    """
    shape = shape or rng.choice(SHAPES)
    if shape == 'pure_turn':
        # Rotation only. Small turns are the interesting case: the sweep budget
        # is 3 s + deg/12, so a 10 degree request gets barely more time than a
        # 170 degree one gets per degree.
        turn = rng.choice([rng.uniform(-180., 180.), rng.uniform(-25., 25.)])
        return shape, dict(waypoints_cm=[], initial_turn_deg=round(turn, 2))

    goal = sample_point(rng, radius)
    if shape == 'direct':
        # No initial turn: the executor must work out the pivot itself.
        points = [goal]
        turn = 0.
    elif shape == 'turn_then_go':
        # Face the goal first, then the waypoint expressed in the *submission*
        # frame -- which is still the pre-turn frame, so this also exercises
        # the pivot-shift bookkeeping.
        points = [goal]
        turn = round(bearing_to(goal), 2)
    else:                                   # polyline
        # Resample rather than shrink: a rejected draw keeps the vertices
        # uniform over the disc, where clamping would pile them up near the
        # origin and quietly stop testing the wide corners.
        for _ in range(200):
            points = [sample_point(rng, radius)
                      for _ in range(rng.randint(1, 3))] + [goal]
            if path_length(dict(waypoints_cm=points)) <= max_path_cm:
                break
        else:
            points = [goal]                 # disc too large to wander inside
        turn = 0.
    return shape, dict(waypoints_cm=[[round(v, 2) for v in p] for p in points],
                       initial_turn_deg=turn)


def narrate(root, heading, body):
    """Append one section to the run's history, which is what the viz renders.

    mission_log_server reads mission-history.md and splits it on '## ' headings,
    so anything written here shows up newest-first on the dashboard. Flushed,
    never fsync'd -- per-record fsync is what destroyed the previous SD card.
    """
    import datetime
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(os.path.join(root, 'mission-history.md'), 'a') as handle:
        handle.write('\n## %s %s\n\n%s\n' % (heading, stamp, body))
        handle.flush()


def residual(result):
    """How far the robot still is from the last waypoint it was given.

    `remaining_trajectory_cm` is whatever was not reached, already rebased into
    the final pose, so the last entry is the miss. An empty list means every
    waypoint was reached and the miss is zero by construction.
    """
    remaining = result.get('remaining_trajectory_cm') or []
    return math.hypot(*remaining[-1]) if remaining else 0.


def path_length(trajectory):
    total, last = 0., [0., 0.]
    for point in trajectory['waypoints_cm']:
        total += math.hypot(point[0] - last[0], point[1] - last[1])
        last = point
    return total


def wait_for_terminal(execution_id, limit=180.):
    """Poll until *this* execution stops, returning its final result.

    Keyed on the execution id on purpose. `/api/local-execution` always answers
    with the most recent job, so a submission that never started leaves the
    previous job's terminal result sitting there -- and polling for "any
    terminal phase" reads that as a pass. Fourteen randomised turns scored
    14/14 that way while only four of them had actually moved the robot.
    """
    deadline = time.monotonic() + limit
    last = {}
    while time.monotonic() < deadline:
        try:
            last = get('/api/local-execution')
        except (urllib.error.URLError, OSError) as exc:
            last = {'phase': 'unreachable', 'error': str(exc)}
        if last.get('execution_id') == execution_id and last.get('phase') in TERMINAL:
            return last
        time.sleep(.4)
    return dict(last, phase='harness-timeout',
                error='no terminal phase for execution %s within %.0f s'
                      % (execution_id, limit))


def wait_until_free(limit=60.):
    """Block while the controller still owns the robot.

    An execution stays `running` for a moment after its last motion while it
    writes artifacts, and a submission during that window is refused outright.
    """
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        try:
            if not get('/api/progress').get('running'):
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(.3)
    return False


def execute(trajectory, extra=None):
    """Capture a fresh frame, submit the trajectory, wait for it to finish."""
    if not wait_until_free():
        return dict(phase='rejected', error='controller never went idle')
    state = post('/api/capture', {})
    token = state.get('frame_token')
    body = dict(trajectory=trajectory, frame_token=token)
    body.update(extra or {})
    try:
        answer = post('/api/local-execution', body)
    except urllib.error.HTTPError as exc:
        return dict(phase='rejected', error=exc.read().decode()[:400],
                    frame_token=token)
    except (urllib.error.URLError, OSError) as exc:
        return dict(phase='rejected', error=str(exc), frame_token=token)
    # A refusal answers with bench state rather than a job, so there is no id.
    # Treat that as the failure it is instead of falling through to the poll.
    started = answer.get('execution_id')
    if not started:
        return dict(phase='rejected', frame_token=token,
                    error='planner did not start an execution: '
                          + str(answer.get('error')
                                or (answer.get('log') or '').strip().splitlines()[-1:]
                                or answer)[:300])
    result = wait_for_terminal(started)
    return dict(result, frame_token=token)


def go_home(log):
    """Drive back to where the run started, then square up the heading.

    Uses the executor's own accumulated run pose, so it inherits whatever drift
    the run has collected. That is the point: a trial that starts from a badly
    estimated origin is still a fair trial of the executor, and the recorded
    residual says how far the estimate has slipped.
    """
    import fetch
    state = get('/api/local-execution')
    pose = state.get('run_pose_cm_deg') or [0., 0., 0.]
    home = fetch.rebase([[0., 0.]], pose)[0]
    moved = {}
    if math.hypot(*home) > 4.:
        moved = execute(dict(waypoints_cm=[[round(home[0], 2), round(home[1], 2)]],
                             initial_turn_deg=0.))
        log('  return leg -> %s %s' % (moved.get('phase'), moved.get('error') or ''))
        state = get('/api/local-execution')
        pose = state.get('run_pose_cm_deg') or [0., 0., 0.]
    square = -((pose[2] + 180.) % 360. - 180.)
    if abs(square) > 3.:
        turned = execute(dict(waypoints_cm=[], initial_turn_deg=round(square, 2)))
        log('  square up -> %s %s' % (turned.get('phase'), turned.get('error') or ''))
    final = get('/api/local-execution').get('run_pose_cm_deg') or [0., 0., 0.]
    log('  back at %.1f, %.1f cm  %.1f deg from run start'
        % (final[0], final[1], final[2]))
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--radius', type=float, default=100.,
                        help='sampling radius in cm; the claim under test is 100')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--shape', choices=SHAPES, default=None,
                        help='force one shape instead of sampling them')
    parser.add_argument('--max-path-cm', type=float, default=150.,
                        help='cap on driven path length, so the executor 90 s '
                             'limit is not mistaken for an unreachable goal')
    parser.add_argument('--dry-run', action='store_true',
                        help='generate and validate only; no motion, no planner')
    parser.add_argument('--no-home', action='store_true',
                        help='skip the return-to-origin leg between trials')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    rng = random.Random(seed)

    import trajectory_executor

    if args.dry_run:
        out = sys.stdout
        record = None
    else:
        if not os.path.ismount('/mnt/robotlogs'):
            raise SystemExit('robot log volume not mounted')
        with open('/mnt/robotlogs/current-search.json') as handle:
            root = json.load(handle)['root']
        record = open(os.path.join(root, 'local-reach-trials.jsonl'), 'a')
        out = sys.stdout

    def log(line):
        out.write(line + '\n')
        out.flush()

    log('seed %d | %d trial(s) | radius %.0f cm | shape %s'
        % (seed, args.trials, args.radius, args.shape or 'sampled'))

    results = []
    for index in range(args.trials):
        shape, trajectory = sample_trajectory(rng, args.radius, args.shape,
                                              args.max_path_cm)
        # Validate against the executor's own rules before sending anything, so
        # a generator bug is never reported as an executor failure.
        try:
            trajectory_executor.validate(dict(trajectory=trajectory))
        except Exception as exc:
            log('%2d %-13s GENERATOR PRODUCED AN INVALID TRAJECTORY: %s'
                % (index, shape, exc))
            continue
        goal = (trajectory['waypoints_cm'][-1] if trajectory['waypoints_cm']
                else [0., 0.])
        summary = ('%2d %-13s turn %7.2f deg | %d point(s) | %5.1f cm path | '
                   'goal %6.1f,%6.1f (%5.1f cm, %6.1f deg)'
                   % (index, shape, trajectory['initial_turn_deg'],
                      len(trajectory['waypoints_cm']), path_length(trajectory),
                      goal[0], goal[1], math.hypot(*goal), bearing_to(goal)))
        if args.dry_run:
            log(summary)
            results.append(dict(index=index, shape=shape, trajectory=trajectory))
            continue

        log(summary)
        started = time.time()
        result = execute(trajectory)
        phase, error = result.get('phase'), result.get('error')
        log('   -> %s%s  (%.1f s, %.1f cm driven)'
            % (phase, ': ' + error if error else '', time.time() - started,
               result.get('travelled_cm') or 0.))

        entry = dict(index=index, seed=seed, shape=shape, trajectory=trajectory,
                     phase=phase, error=error,
                     travelled_cm=result.get('travelled_cm'),
                     pose_cm_deg=result.get('pose_cm_deg'),
                     completed_waypoints=result.get('completed_waypoints'),
                     tracking_rejections=result.get('tracking_rejections'),
                     execution_id=result.get('execution_id'),
                     residual_cm=residual(result),
                     at=time.time())
        results.append(entry)
        record.write(json.dumps(entry) + '\n')
        record.flush()
        narrate(root, 'Trial %d (%s)' % (index, shape),
                '%s\n\n```json\n%s\n```\n\nResult: **%s**%s, %.1f cm driven, '
                '%d/%d waypoint(s), %d tracking rejection(s), %.1f cm residual.'
                % (summary.strip(), json.dumps(trajectory), phase,
                   ' — ' + error if error else '', entry['travelled_cm'] or 0.,
                   entry['completed_waypoints'] or 0,
                   len(trajectory['waypoints_cm']),
                   entry['tracking_rejections'] or 0, entry['residual_cm']))

        if not args.no_home:
            go_home(log)

    if record:
        record.close()

    if not args.dry_run:
        log('')
        log('--- summary (seed %d) ---' % seed)
        tally = {}
        for entry in results:
            tally.setdefault(entry['phase'], []).append(entry)
        for phase in sorted(tally):
            log('%-10s %d' % (phase, len(tally[phase])))
        for entry in results:
            if entry['phase'] in ('failed', 'rejected'):
                log('  %-13s %s' % (entry['shape'], entry['error']))


if __name__ == '__main__':
    main()
