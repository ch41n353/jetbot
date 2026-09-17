"""Steering sign, margin and output bounds; no robot or rendered simulation."""
import math
import os
import sys
import unittest
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
from route_executor import straight_steering


class RouteSteeringTests(unittest.TestCase):
    def test_centered_straight_motion_has_no_differential_command(self):
        self.assertEqual(straight_steering(0.,0.,0.,1),0.)

    def test_lateral_correction_cannot_demand_heading_near_guard(self):
        # Even a large position error stops demanding rotation at +/-1.5deg.
        self.assertAlmostEqual(straight_steering(100.,math.radians(-1.5),0.,1),0.)
        self.assertAlmostEqual(straight_steering(-100.,math.radians(1.5),0.,1),0.)
        self.assertLess(straight_steering(.5,0.,0.,1),0.)

    def test_rate_damping_opposes_rotation_at_desired_heading(self):
        self.assertLess(straight_steering(0.,0.,math.radians(20),1),0.)
        self.assertGreater(straight_steering(0.,0.,math.radians(-20),1),0.)

    def test_recorded_prebrake_state_now_countersteers(self):
        # Actual last powered state before the -5.17deg coast failure.
        x,yaw,rate=.8824146,math.radians(-2.64),math.radians(-16.3)
        old=max(-.025,min(.025,-.3*yaw-.02*x))
        self.assertLess(old,0.)
        self.assertGreater(straight_steering(x,yaw,rate,1),0.)

    def test_forward_reverse_and_left_right_are_symmetric(self):
        self.assertAlmostEqual(straight_steering(.5,.01,.1,1),
                               -straight_steering(-.5,-.01,-.1,1))
        self.assertAlmostEqual(straight_steering(.5,.01,.1,1),
                               straight_steering(-.5,.01,.1,-1))

    def test_output_limits_and_invalid_input(self):
        for direction in (-1,1):
            for x in (-10.,0.,10.):
                for yaw in (-.5,0.,.5):
                    for rate in (-2.,0.,2.):
                        self.assertLessEqual(abs(straight_steering(x,yaw,rate,direction)),.025)
        with self.assertRaises(ValueError):straight_steering(float('nan'),0.,0.,1)
        with self.assertRaises(ValueError):straight_steering(0.,0.,0.,0)


if __name__=='__main__':unittest.main()
