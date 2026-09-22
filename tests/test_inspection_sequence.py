import json
import threading
import unittest
import urllib.request
from unittest.mock import patch
from test_planner_motion_contract import FakeRobot, fetch, planner_demo
import inspection_sequence as sequence

class InspectionTests(unittest.TestCase):
    def test_http_four_way_returns_four_views_without_model_calls(self):
        robot = FakeRobot()
        robot.check = lambda: None
        turns = []
        robot.turn = lambda angle: turns.append(angle) or angle
        with patch.object(fetch, 'Robot', return_value=robot), patch.object(fetch, 'Odometer', return_value=object()), patch.object(fetch, 'call', return_value={'power':{}, 'imu':{}}), patch.object(fetch, 'recognize') as model:
            bench = planner_demo.Bench(False)
            server = planner_demo.Server(('127.0.0.1', 0), planner_demo.Handler)
            server.bench = bench; server.open_lan = False
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            try:
                url = 'http://127.0.0.1:%d/api/inspection' % server.server_address[1]
                req = urllib.request.Request(url, data=b'{"preset":"four_way"}', headers={'Content-Type':'application/json'})
                self.assertTrue(json.load(urllib.request.urlopen(req))['inspection'])
                bench.flight.join(5)
                result = json.load(urllib.request.urlopen(url))
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(turns, [90,90,90])
                self.assertEqual([x['relative_heading_degrees'] for x in result['images']], [0,90,180,270])
                self.assertTrue(result['contact_sheet_base64'])
                model.assert_not_called()
                self.assertTrue(all(x == (0,0) for x in robot.outputs))
            finally:
                server.shutdown(); server.server_close(); thread.join(2)

    def test_invalid_batch_rejected_before_execution(self):
        for body in [{'steps':[{'action':'turn','degrees':float('nan')}]}, {'steps':[{'action':'drive'}]}, {'steps':[{'action':'turn','degrees':90}]}]:
            with self.assertRaises(ValueError): sequence.validate(body)

    def test_cancel_preserves_partial_images_and_stops(self):
        robot = FakeRobot(); robot.check = lambda: None
        with patch.object(fetch,'Robot',return_value=robot),patch.object(fetch,'Odometer',return_value=object()),patch.object(fetch,'call',return_value={'power':{},'imu':{}}):
            bench = planner_demo.Bench(False)
            def turn(angle):
                bench.abort.set()
                return angle
            robot.turn = turn
            result = dict(status='running', images=[], events=[], start_heading=bench.heading)
            sequence.run(bench, sequence.validate({'preset':'four_way'}), result)
            self.assertEqual(result['status'],'cancelled')
            self.assertEqual(len(result['images']),1)
            self.assertEqual(robot.outputs[-1],(0,0))

    def test_route_failure_stops_before_next_capture(self):
        robot = FakeRobot(); robot.check = lambda: None
        with patch.object(fetch,'Robot',return_value=robot),patch.object(fetch,'Odometer',return_value=object()),patch.object(fetch,'call',return_value={'power':{},'imu':{}}):
            bench = planner_demo.Bench(False)
            steps = sequence.validate({'steps':[{'action':'route','points_cm':[[0,30]],'obstacles_cm':[{'label':'block','point_cm':[20,20]}]}, {'action':'capture'}]})
            for moved, expected in [(0,'failed'), (30,'complete')]:
                result = dict(status='running',images=[],events=[],start_heading=0)
                with patch.object(fetch,'follow',return_value=[0,moved,0]) as follow:
                    sequence.run(bench,steps,result)
                    self.assertEqual(follow.call_args[0][4],[('block',[20.,20.])])
                self.assertEqual(result['status'],expected)
                self.assertEqual(len(result['images']),int(expected=='complete'))

if __name__ == '__main__': unittest.main()
