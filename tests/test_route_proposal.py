"""Offline checks for model route proposals. No images, calibration or motors."""
import math
import os
import sys
import unittest
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
from route_proposal import validate_route,validate_answer

# Synthetic pinhole floor: v=200 is the horizon, a table top hides some floor.
TABLE=(400,500,300,340)


def project(point):
    u,v=point
    if v<220:raise ValueError('above the floor line')
    if TABLE[0]<=u<=TABLE[1] and TABLE[2]<=v<=TABLE[3]:raise ValueError('table top, not floor')
    forward=3000./(v-200.)
    return ((u-320.)*forward/300.,forward)


def pixel(right,forward):
    return dict(x=320.+right*300./forward,y=200.+3000./forward)


def clear(start,end):
    return True


def blocked(start,end):
    # One obstacle footprint straddling 45-65 cm ahead on the centre line.
    return not (45<=end[1]<=65 and abs(end[0])<10)


class RouteProposalTests(unittest.TestCase):
    def test_clean_route_accepted(self):
        route=[pixel(0,15),pixel(5,30),pixel(-5,50),pixel(0,70)]
        result=validate_route(route,project,clear)
        self.assertEqual(result['status'],'accepted')
        self.assertEqual(len(result['waypoints_cm']),4)
        self.assertFalse(any(w['subdivided'] for w in result['waypoints_cm']))
        self.assertEqual([w['pixel_index'] for w in result['waypoints_cm']],[0,1,2,3])
        last=result['waypoints_cm'][-1]
        self.assertAlmostEqual(last['forward_cm'],70)
        self.assertAlmostEqual(last['right_cm'],0)
        self.assertLess(abs(result['path_length_cm']-70),12)

    def test_point_off_the_floor_is_rejected(self):
        for bad,index in [([pixel(0,15),dict(x=320,y=210)],1),
                          ([pixel(0,15),dict(x=450,y=320),pixel(0,60)],1)]:
            result=validate_route(bad,project,clear)
            self.assertEqual(result['reason'],'pixel_not_on_floor')
            self.assertEqual(result['index'],index)
            self.assertEqual(result['waypoints_cm'],[])

    def test_pixel_outside_frame_is_rejected(self):
        result=validate_route([pixel(0,15),dict(x=700,y=300)],project,clear)
        self.assertEqual((result['status'],result['reason'],result['index']),
                         ('rejected','pixel_outside_frame',1))

    def test_doubling_back_is_rejected(self):
        result=validate_route([pixel(0,20),pixel(0,45),pixel(0,30)],project,clear)
        self.assertEqual(result['reason'],'route_doubles_back')
        self.assertEqual(result['index'],2)

    def test_blocked_segment_names_the_failing_index(self):
        route=[pixel(0,20),pixel(0,40),pixel(0,60),pixel(0,80)]
        self.assertEqual(validate_route(route,project,clear)['status'],'accepted')
        result=validate_route(route,project,blocked)
        self.assertEqual(result['reason'],'segment_blocked')
        self.assertEqual(result['index'],2)
        self.assertEqual(result['waypoint_index'],2)
        self.assertEqual(result['waypoints_cm'],[])

    def test_long_gap_is_subdivided_and_every_piece_checked(self):
        checked=[]
        def record(start,end):
            checked.append((start,end));return True
        result=validate_route([pixel(0,60)],project,record)
        self.assertEqual(result['status'],'accepted')
        forwards=[w['forward_cm'] for w in result['waypoints_cm']]
        self.assertEqual(len(forwards),3)
        for expected,actual in zip([20,40,60],forwards):self.assertAlmostEqual(actual,expected)
        self.assertEqual([w['subdivided'] for w in result['waypoints_cm']],[True,True,False])
        self.assertEqual(len(checked),3)
        self.assertAlmostEqual(checked[0][0][1],0)
        self.assertTrue(all(math.hypot(e[0]-s[0],e[1]-s[1])<=25.0001 for s,e in checked))

    def test_over_length_route_is_rejected(self):
        route=[pixel(0,25),pixel(20,50),pixel(-20,75),pixel(20,100)]
        result=validate_route(route,project,clear)
        self.assertEqual(result['reason'],'route_too_long')
        self.assertEqual(result['index'],3)
        self.assertEqual(validate_route(route[:3],project,clear)['status'],'accepted')

    def test_range_and_spacing_bounds(self):
        far=validate_route([pixel(0,40),pixel(0,145)],project,clear)
        self.assertEqual((far['reason'],far['index']),('beyond_trusted_range',1))
        # Perspective makes evenly spaced pixels uneven on the ground, so a
        # crowding point is dropped and reported rather than failing the route.
        near=validate_route([pixel(0,15),pixel(0,18)],project,clear)
        self.assertEqual(near['status'],'accepted')
        self.assertEqual([d['pixel_index'] for d in near['dropped_points']],[0])
        self.assertEqual([round(w['forward_cm'],1) for w in near['waypoints_cm']],[18.])
        # A middle point that crowds its predecessor gives way; the target stays.
        mid=validate_route([pixel(0,15),pixel(0,17),pixel(0,40)],project,clear)
        self.assertEqual(mid['status'],'accepted')
        self.assertEqual([d['pixel_index'] for d in mid['dropped_points']],[1])
        self.assertEqual(mid['waypoints_cm'][-1]['pixel_index'],2)
        # The robot's own position never counts as a crowding predecessor.
        first=validate_route([pixel(0,20)],lambda p:(0.,3.),clear)
        self.assertEqual(first['status'],'accepted')

    def test_empty_route_is_not_a_rejection(self):
        result=validate_route([],project,clear)
        self.assertEqual((result['status'],result['reason']),('empty','no_route_proposed'))
        self.assertEqual(result['waypoints_cm'],[])
        self.assertEqual(validate_answer(dict(target_visible=False),project,clear)['status'],'empty')

    def test_malformed_input_is_rejected(self):
        for route in (None,'route',42,dict(x=1,y=2)):
            self.assertEqual(validate_route(route,project,clear)['reason'],'malformed_route')
        for route in ([dict(x='a',y=300)],[dict(x=320)],[None],['nope'],
                      [dict(x=float('nan'),y=300)],[dict(x=320,y=True)],[[320,300,1]]):
            result=validate_route(route,project,clear)
            self.assertEqual((result['reason'],result['index']),('non_numeric_pixel',0))
        self.assertEqual(validate_answer('not an answer',project,clear)['reason'],'malformed_route')

    def test_callable_failures_and_limits_fail_closed(self):
        def raises(start,end):raise ValueError('map missing')
        result=validate_route([pixel(0,20)],project,raises)
        self.assertEqual(result['reason'],'clearance_check_failed')
        self.assertEqual(validate_route([pixel(0,20)],lambda p:None,clear)['reason'],'pixel_not_on_floor')
        self.assertEqual(validate_route([pixel(0,20)],lambda p:(0,-5),clear)['reason'],'pixel_not_on_floor')
        self.assertEqual(validate_route([pixel(0,20)]*30,project,clear)['reason'],'too_many_points')
        for bad in (dict(min_spacing_cm=0),dict(min_spacing_cm=20,max_spacing_cm=25),
                    dict(max_range_cm=float('nan')),dict(max_points=2.5)):
            with self.assertRaises(ValueError):validate_route([pixel(0,20)],project,clear,**bad)

    def test_answer_route_is_validated_and_limits_configurable(self):
        answer=dict(target_visible=True,route_pixels=[pixel(0,20),pixel(0,40)])
        self.assertEqual(validate_answer(answer,project,clear)['status'],'accepted')
        self.assertEqual(validate_answer(answer,project,clear,max_path_cm=30.)['reason'],'route_too_long')
        tight=validate_answer(answer,project,clear,max_spacing_cm=10.,min_spacing_cm=5.)
        self.assertEqual(len(tight['waypoints_cm']),4)


if __name__=='__main__':unittest.main()
