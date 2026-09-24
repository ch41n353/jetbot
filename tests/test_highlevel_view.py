import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from highlevel_view import view


class HighLevelViewTests(unittest.TestCase):
    def test_reset_context_is_blank_before_first_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cv2.imwrite(str(root/'readiness.jpg'),np.zeros((480,640,3),np.uint8))
            result=view(root)
            self.assertFalse(result['available'])
            self.assertIn('No image has been reviewed',result['note'])
    def test_renders_exact_supplied_reference_and_goal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cv2.imwrite(str(root / 'handoff.jpg'), np.zeros((480, 640, 3), np.uint8))
            visual = dict(
                frame_token=17,
                route_pixels=[dict(x=320, y=450), dict(x=180, y=330)],
                goal_pixel=dict(x=90, y=210),
            )
            command = dict(
                reviewed_at='2026-09-23T20:00:00+00:00',
                prompt='Approach the visible bottle',
                review_image='handoff.jpg',
                high_level_visual=visual,
            )
            (root / 'highlevel-review.json').write_text(json.dumps(command))

            result = view(root)

            self.assertTrue(result['available'])
            self.assertEqual(result['goal'], visual['goal_pixel'])
            self.assertEqual(result['trajectory_pixels'], visual['route_pixels'])
            self.assertNotIn('No reference trajectory', result['note'])
            self.assertTrue(result['image'].startswith('data:image/jpeg;base64,'))

    def test_above_horizon_goal_does_not_hide_rgb_handoff(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cv2.imwrite(str(root / 'handoff.jpg'), np.zeros((480, 640, 3), np.uint8))
            visual=dict(route_pixels=[dict(x=320,y=420),dict(x=300,y=220)],goal_pixel=dict(x=300,y=180))
            command=dict(reviewed_at='2026-09-23T21:00:00+00:00',prompt='Distant can',
                         review_image='handoff.jpg',high_level_visual=visual)
            (root/'highlevel-review.json').write_text(json.dumps(command))
            result=view(root)
            self.assertTrue(result['available'])
            self.assertEqual(result['trajectory_pixels'],visual['route_pixels'])
            self.assertEqual(result['goal'],visual['goal_pixel'])
            self.assertTrue(result['image'].startswith('data:image/jpeg;base64,'))

    def test_new_handoff_does_not_replace_reviewed_image(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            cv2.imwrite(str(root/'reviewed.jpg'),np.zeros((480,640,3),np.uint8))
            cv2.imwrite(str(root/'later.jpg'),np.full((480,640,3),255,np.uint8))
            record=dict(reviewed_at='one',prompt='reviewed',review_image='reviewed.jpg',high_level_visual={})
            (root/'highlevel-review.json').write_text(json.dumps(record))
            (root/'commands.jsonl').write_text(json.dumps(dict(issued_at='two',url='/api/mission',handoff_image='later.jpg'))+'\n')
            result=view(root)
            self.assertEqual(result['prompt'],'reviewed')
            self.assertEqual(result['issued_at'],'one')


if __name__ == '__main__':
    unittest.main()
