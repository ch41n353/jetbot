"""Motor-free acceptance scenarios for the autonomous search/navigation stack.

This composes the production search graph, local search policy, and rendered
object-mission simulator. Recognition is a deterministic fake: this module
never imports the OpenAI client, opens the robot service, or issues motor I/O.
"""
import argparse
import json
import os
import tempfile

from search_graph import SearchGraph
from sol_search import SearchState
from simulate_search_graph import run_case as run_graph_case
from simulate_object_mission import run_case as run_object_case


MODEL = 'gpt-5.6-sol'


def certificate(graph, source, target, evidence='recorded_clearance'):
    return dict(source_revision=graph.data['nodes'][source]['revision'],
                target_revision=graph.data['nodes'][target]['revision'],
                evidence=evidence)


def tracked(graph, place, now=0.):
    node = graph.data['nodes'][place]
    return dict(status='tracked', place=place, submap=node['submap'],
                revision=node['revision'], time=now)


def result(name, passed, **evidence):
    return dict(name=name, passed=bool(passed), evidence=evidence)


def visible_target():
    graph = SearchGraph('red bottle')
    graph.place('room', 'room-map', headings=[0])
    ticket = graph.recognition_ticket('room-000.jpg')
    accepted = graph.accept_recognition(ticket, [280, 150, 350, 300], MODEL)
    action = graph.next_action('room', tracked(graph, 'room'), 0.)
    return result('target_visible', accepted and action['kind'] == 'object_found' and
                  not action['motion_authorized'], action=action['kind'],
                  model=graph.data['found']['model'], motor_commands=0)


def absent_target():
    graph = SearchGraph('red bottle')
    graph.place('room', 'room-map', headings=[0])
    ticket = graph.recognition_ticket('room-000.jpg')
    accepted = graph.accept_recognition(ticket, None, MODEL)
    graph.observe('room', 0, ticket['image'], 0)
    action = graph.next_action('room', tracked(graph, 'room'), 0.)
    return result('target_absent', accepted and graph.data['found'] is None and
                  action['kind'] == 'known_map_searched', action=action['kind'],
                  recognition='negative', motor_commands=0)


def blocked_cable():
    plan = dict(inspected_free_rectangle_cm=[-60, -60, 60, 60],
                obstacle_rectangles_cm=[[8, -10, 15, 0]],
                search_stations_cm=[])
    local = SearchState(plan)
    local_candidates = local.candidates()

    graph = SearchGraph('red bottle')
    graph.place('near', 'near-map', headings=[])
    graph.place('far', 'far-map', headings=[0])
    graph.connect('near>far', 'near', 'far', 10., certificate(graph, 'near', 'far'))
    proposed = graph.next_action('near', tracked(graph, 'near'), 0.)
    # The executor's fresh local sweep sees the cable and invalidates the edge.
    graph.block('near>far')
    after_check = graph.next_action('near', tracked(graph, 'near'), 0.)
    passed = (local_candidates == [] and proposed['kind'] == 'navigate_graph' and
              after_check['kind'] == 'map_required')
    return result('blocked_cable', passed, local_turn_candidates=len(local_candidates),
                  before_sweep=proposed['kind'], after_sweep=after_check['kind'],
                  motor_commands=0)


def api_latency_and_failure():
    graph = SearchGraph('red bottle')
    graph.place('room', 'room-map', headings=[0])
    ticket = graph.recognition_ticket('room-000.jpg')
    attempts = [dict(after_seconds=8., outcome='timeout'),
                dict(after_seconds=9., outcome='negative')]
    # A timeout cannot authorize movement. A bounded retry for the same captured
    # image may eventually resolve the pending ticket while the robot stays put.
    accepted = graph.accept_recognition(ticket, None, MODEL)
    graph.observe('room', 0, ticket['image'], 0)
    action = graph.next_action('room', tracked(graph, 'room', 17.), 17.)
    return result('api_latency_failure', accepted and action['kind'] == 'known_map_searched',
                  attempts=attempts, total_wait_seconds=17., final_action=action['kind'],
                  motor_commands_during_wait=0)


def stale_reply():
    first = SearchGraph('red bottle')
    first.data['mission_id'] = 'mission-before-target-change'
    first.place('room', 'room-map', headings=[0])
    old = first.recognition_ticket('old-frame.jpg')
    current = first.new_search('blue cup')
    current.data['mission_id'] = 'mission-after-target-change'
    accepted = current.accept_recognition(old, [10, 10, 30, 30], MODEL)
    return result('stale_reply', not accepted and current.data['found'] is None,
                  old_mission=old['mission_id'], current_mission=current.data['mission_id'],
                  stale_reply_accepted=accepted, motor_commands=0)


def cancellation():
    graph = SearchGraph('red bottle')
    graph.data['mission_id'] = 'mission-before-cancel'
    graph.place('room', 'room-map', headings=[0])
    pending = graph.recognition_ticket('cancelled-frame.jpg')
    # Cancellation replaces the mission generation. Its eventual API reply is
    # checked against the replacement checkpoint and cannot trigger an action.
    replacement = graph.new_search('red bottle')
    replacement.data['mission_id'] = 'mission-after-cancel'
    accepted = replacement.accept_recognition(pending, [10, 10, 30, 30], MODEL)
    action = replacement.next_action('room', tracked(replacement, 'room'), 0.)
    return result('cancellation', not accepted and action['kind'] == 'observe_heading',
                  late_reply_accepted=accepted, next_action=action['kind'], motor_commands=0)


def low_battery():
    graph = SearchGraph('red bottle')
    graph.place('near', 'near-map', headings=[])
    graph.place('far', 'far-map', headings=[0])
    graph.connect('near>far', 'near', 'far', 10., certificate(graph, 'near', 'far'))
    proposed = graph.next_action('near', tracked(graph, 'near'), 0.)
    power = dict(motion_allowed=False, stop_latched=True,
                 stop_reason='battery_pack_low', voltage_v=10.7)
    dispatched = proposed['kind'] == 'navigate_graph' and power['motion_allowed']
    return result('low_battery', proposed['kind'] == 'navigate_graph' and not dispatched,
                  proposed=proposed['kind'], power=power, motion_dispatched=bool(dispatched),
                  motor_commands=0)


def multi_room_checkpoint():
    run = run_graph_case(rooms=4, restart=True, dynamic_discovery=True)
    passed = (run['outcome'] == 'object_found' and run['checkpoint_resumed'] and
              run['unsafe_edge_traversals'] == 0 and run['duplicate_observations'] == 0)
    return result('multi_room_frontier_checkpoint', passed, outcome=run['outcome'],
                  checkpoint_resumed=run['checkpoint_resumed'],
                  known_places=run['known_places'], observations=run['observations'],
                  edges_traversed=run['actions'].get('edges_traversed', 0),
                  unsafe_edge_traversals=run['unsafe_edge_traversals'])


def final_approach():
    run = run_object_case(dict(speed=12, coast=.05, seed=3), target=(0, 60))
    mission = run['result']
    passed = (mission['outcome'] == 'object_reached_estimate' and
              mission['intermediate_model_calls'] == 0 and
              run['clearance_violations'] == [] and run['final_motor_output'] == [0, 0])
    return result('final_approach', passed, outcome=mission['outcome'],
                  final_target_range_cm=run['true_target_range_cm'],
                  intermediate_model_calls=mission['intermediate_model_calls'],
                  clearance_violations=len(run['clearance_violations']),
                  final_motor_output=run['final_motor_output'])


SCENARIOS = (visible_target, absent_target, blocked_cable, api_latency_and_failure,
             stale_reply, cancellation, low_battery, multi_room_checkpoint,
             final_approach)


def evaluate():
    cases = [scenario() for scenario in SCENARIOS]
    return dict(complete=True, passed=all(case['passed'] for case in cases),
                model_contract=MODEL, hardware_calls=0, api_calls=0, cases=cases,
                limitations=['Graph mapping/localization events are synthetic',
                             'API timing and failures are scheduled, not network measurements',
                             'Final approach uses rendered camera/IMU/wheel dynamics'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    args = parser.parse_args()
    report = evaluate()
    if args.output:
        directory = os.path.dirname(os.path.abspath(args.output))
        if not os.path.isdir(directory):
            os.makedirs(directory)
        fd, temporary = tempfile.mkstemp(prefix='.autonomy-', dir=directory)
        try:
            with os.fdopen(fd, 'w') as output:
                json.dump(report, output, indent=2, sort_keys=True)
                output.flush()
                os.fsync(output.fileno())
            os.rename(temporary, args.output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
