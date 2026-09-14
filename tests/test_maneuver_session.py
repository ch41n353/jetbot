import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_nav'))
import maneuver_session


class SessionTests(unittest.TestCase):
    def test_batch_dispatch_uses_cancellable_worker(self):
        command = maneuver_session.route_command('plan.json', 'result.json', True, 125, batch=True)
        self.assertTrue(command[1].endswith('approach_batch.py'))
        self.assertNotIn('--predictive-braking', command)
        self.assertIn('--execute', command)

    def run_session(self, armed, commands, power_allowed=True):
        created, operations, emitted = [], [], []

        class Process:
            def __init__(self, command):
                self.command, self.returncode = command, None
            def poll(self):
                return self.returncode
            def terminate(self):
                operations.append('terminate_worker' if len(created) > 1 and self is created[1] else 'terminate_service')
                self.returncode = -15
            def wait(self, timeout=None):
                self.returncode = 0 if self.returncode is None else self.returncode
                return self.returncode

        def spawn(command, **kwargs):
            process = Process(command)
            created.append(process)
            return process

        def call(action, **fields):
            operations.append(action)
            if action == 'shutdown':
                created[0].returncode = 0
            return dict(healthy=True, motion_enabled=armed, session_id='owned',
                        power=dict(motion_allowed=power_allowed,voltage_v=5.04 if power_allowed else 4.7))

        original_exists = os.path.exists
        with tempfile.TemporaryDirectory() as directory:
            argv = ['maneuver_session.py', '--output-directory', directory] + (['--enable-motion'] if armed else [])
            plan = os.path.join(directory, 'plan.json')
            with open(plan, 'w') as out:
                json.dump({'captured_monotonic': 0}, out)
            stdin = io.StringIO(commands.replace('/tmp/plan.json', plan))
            with patch.object(sys, 'argv', argv), patch.object(sys, 'stdin', stdin), \
                    patch.object(stdin, 'fileno', return_value=0), \
                    patch('maneuver_session.os.read', side_effect=[stdin.getvalue().encode(), b'']), \
                    patch('maneuver_session.select.select', return_value=([stdin], [], [])), \
                    patch('maneuver_session.subprocess.Popen', side_effect=spawn), \
                    patch('maneuver_session.call', side_effect=call), \
                    patch('maneuver_session.emit', side_effect=lambda **event: emitted.append(event)), \
                    patch('maneuver_session.os.path.exists', side_effect=lambda path: False if path == '/tmp/jetbot-local-nav/control.sock' else original_exists(path)):
                maneuver_session.main()
        return created, operations, emitted

    def test_disarmed_session_cannot_launch_route(self):
        created, operations, emitted = self.run_session(False, '{"command":"execute","plan":"/tmp/plan.json"}\n{"command":"shutdown"}\n')
        self.assertEqual(len(created), 1)
        self.assertNotIn('--enable-motion', created[0].command)
        self.assertTrue(any(e.get('reason') == 'Session is disarmed' for e in emitted))
        self.assertIn('shutdown', operations)

    def test_power_fault_prevents_new_route_and_emits_event(self):
        created,operations,emitted=self.run_session(True,'{"command":"execute","plan":"/tmp/plan.json"}\n{"command":"shutdown"}\n',False)
        self.assertEqual(len(created),1)
        self.assertTrue(any(e.get('event')=='power_stopped' for e in emitted))
        self.assertTrue(any(e.get('reason')=='Power guard blocks motion' for e in emitted))

    def test_stop_is_handled_while_worker_is_running(self):
        created, operations, emitted = self.run_session(True, '{"command":"execute","plan":"/tmp/plan.json"}\n{"command":"stop"}\n{"command":"shutdown"}\n')
        self.assertEqual(len(created), 2)
        self.assertLess(operations.index('stop'), operations.index('terminate_worker'))
        self.assertIn('shutdown', operations)
        self.assertTrue(any(e['event'] == 'route_finished' for e in emitted))
        self.assertTrue(any(e['event'] == 'session_closed' for e in emitted))
