import os
import sys
import unittest
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
from spatial_executor import drive_group
from simulate_spatial import run_spatial_case


class CoalescedDriveTests(unittest.TestCase):
    def test_group_keeps_direction_turn_and_distance_boundaries(self):
        def actions(*values):
            return [dict(kind='turn' if v=='turn' else 'drive',value=30 if v=='turn' else v) for v in values]
        for values,total,count in [((10,15,10),25,2),((10,-10),10,1),((10,'turn',10),10,1)]:
            pending=actions(*values)
            merged,n=drive_group(pending,True)
            self.assertEqual((merged['value'],n),(total,count))
            self.assertEqual(len(pending),len(values))

    def test_thirty_cm_has_one_powered_run_and_one_final_settle(self):
        case=run_spatial_case(dict(speed=12,coast=.05,seed=3),goal=(0,30),
                              measured_map_drive=True,coalesce_drives=True)
        result=case['result']
        self.assertEqual(result['outcome'],'target_reached_estimate',result.get('reason'))
        self.assertEqual(len(result['actions']),1)
        self.assertEqual(result['primitive_count'],2)
        self.assertLess(case['true_goal_error_cm'],3)
        self.assertEqual(case['clearance_violations'],[])
        commands=case['motor_commands']
        powered=[i for i,c in enumerate(commands) if c['action']=='motors_hold']
        self.assertTrue(powered)
        self.assertFalse(any(c['action']=='stop' for c in commands[powered[0]:powered[-1]]))
        self.assertEqual(case['watchdog_stops'],0)
        self.assertEqual(case['final_motor_output'],[0,0])

    def test_camera_loss_after_old_boundary_stops_extended_drive(self):
        case=run_spatial_case(dict(speed=12,coast=.05,seed=3,frame_fault_after=1.8),goal=(0,30),
                              measured_map_drive=True,coalesce_drives=True)
        self.assertEqual(case['result']['outcome'],'stopped')
        self.assertIn('camera/IMU failure',case['result']['reason'])
        self.assertEqual(case['final_motor_output'],[0,0])
        self.assertEqual(case['clearance_violations'],[])

    def test_target_arrival_still_stops_before_return(self):
        case=run_spatial_case(dict(speed=12,coast=.05,seed=3),goals=[[0,20],[0,0]],
                              free=[-18,-30,18,34],measured_map_drive=True,coalesce_drives=True)
        self.assertEqual(case['result']['outcome'],'target_reached_estimate',case['result'].get('reason'))
        self.assertEqual(len(case['result']['reached_targets']),2)
        self.assertEqual(len(case['result']['actions']),2)
        commands=case['motor_commands']
        forward=next(i for i,c in enumerate(commands) if c['action']=='motors_hold' and c['fields']['left']>0)
        reverse=next(i for i,c in enumerate(commands) if c['action']=='motors_hold' and c['fields']['left']<0)
        self.assertTrue(any(c['action']=='stop' for c in commands[forward:reverse]))

    def test_extended_drive_keeps_four_second_limit(self):
        case=run_spatial_case(dict(speed=6,coast=.05,seed=3),goal=(0,30),
                              measured_map_drive=True,coalesce_drives=True)
        self.assertEqual(case['result']['outcome'],'stopped')
        self.assertIn('4-second powered limit',case['result']['reason'])
        self.assertEqual(case['final_motor_output'],[0,0])

    def test_stop_after_old_boundary_cannot_restart_motor(self):
        case=run_spatial_case(dict(speed=12,coast=.05,seed=3,cancel_after=1.8),goal=(0,30),
                              measured_map_drive=True,coalesce_drives=True)
        self.assertEqual(case['result']['outcome'],'stopped')
        self.assertIn('cancelled',case['result']['reason'])
        self.assertEqual(case['final_motor_output'],[0,0])

    def test_preview_matches_joined_execution_boundaries(self):
        import tempfile
        import time
        import cv2
        import numpy as np
        from spatial_plan import prepare
        from spatial_preview import write_preview
        with tempfile.TemporaryDirectory() as directory:
            image=os.path.join(directory,'anchor.jpg')
            cv2.imwrite(image,np.zeros((480,640,3),dtype=np.uint8))
            capture=dict(image_path=image,captured_monotonic=time.monotonic(),session_id='test',control_epoch=0)
            request=dict(image_path=image,goals_cm=[[0,20],[0,0]],
                         inspected_free_rectangle_cm=[-40,-50,40,60],obstacle_rectangles_cm=[],
                         measured_map_drive=True,coalesce_drives=True)
            plan,_=prepare(capture,request)
            preview=write_preview(plan,os.path.join(directory,'preview.html'))
            self.assertEqual([a['value'] for a in preview['actions']],[20.,-20.])
            with self.assertRaises(ValueError):prepare(capture,dict(request,measured_map_drive=False))


if __name__=='__main__':unittest.main()
