import os,sys,unittest
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'local_nav'))
from turn_controller import turn_command

class TurnTests(unittest.TestCase):
    def test_right_and_left_commands(self):
        self.assertGreater(turn_command(15,0)['left'],0)
        self.assertLess(turn_command(-15,0)['left'],0)
    def test_stop_at_target_and_small_overshoot(self):
        self.assertIsNone(turn_command(15,14))
        self.assertIsNone(turn_command(-15,-17))
    def test_wrong_direction_and_large_overshoot_stop(self):
        for target,angle in [(15,-4),(15,21),(-15,4),(-15,-21)]:
            with self.assertRaises(RuntimeError):turn_command(target,angle)
    def test_invalid_targets(self):
        for target in [0,31,-31,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):turn_command(target,0)
