import os
import sys
import tempfile
import unittest
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
from search_graph import SearchGraph


class GraphTests(unittest.TestCase):
    def tracked(self,graph,node):
        value=graph.data['nodes'][node]
        return dict(status='tracked',place=node,submap=value['submap'],revision=value['revision'],time=10.)

    def link(self,graph,key,source,target,cost=1):
        graph.connect(key,source,target,cost,dict(evidence='synthetic-test-corridor',
            source_revision=graph.data['nodes'][source]['revision'],target_revision=graph.data['nodes'][target]['revision']))

    def test_hundred_room_route_without_global_dead_reckoning(self):
        graph=SearchGraph('robot')
        for i in range(100):graph.place(str(i),'room_%d'%i,headings=[] if i<99 else [0])
        for i in range(99):self.link(graph,str(i),str(i),str(i+1))
        action=graph.next_action('0',self.tracked(graph,'0'),10.)
        self.assertEqual(action['kind'],'navigate_graph')
        self.assertEqual(len(action['edges']),99)
        self.assertEqual(action['goal'],'99')
        self.assertFalse(action['motion_authorized'])

    def test_closed_door_reroutes(self):
        graph=SearchGraph('robot')
        for name in ('a','b','c','d'):graph.place(name,name,headings=[0] if name=='d' else [])
        self.link(graph,'ab','a','b');self.link(graph,'bd','b','d')
        self.link(graph,'ac','a','c',2);self.link(graph,'cd','c','d',2)
        graph.block('bd')
        self.assertEqual(graph.next_action('a',self.tracked(graph,'a'),10.)['edges'],['ac','cd'])

    def test_unknown_door_is_not_free_space(self):
        graph=SearchGraph('robot');graph.place('a','a',headings=[]);graph.place('b','b')
        graph.connect('ab','a','b',1)
        self.assertEqual(graph.next_action('a',self.tracked(graph,'a'),10.)['kind'],'map_required')

    def test_map_revision_invalidates_route_and_coverage(self):
        graph=SearchGraph('robot');graph.place('a','a',headings=[]);graph.place('b','b',headings=[0])
        self.link(graph,'ab','a','b');graph.observe('b',0,'image.jpg',0)
        graph.place('b','b',revision=1,headings=[0])
        self.assertEqual(graph.missing('b'),[0.])
        self.assertEqual(graph.next_action('a',self.tracked(graph,'a'),10.)['kind'],'map_required')

    def test_checkpoint_resume_preserves_search_and_frontier(self):
        graph=SearchGraph('robot');graph.place('a','room',kind='frontier',headings=[0,30])
        graph.observe('a',0,'first.jpg',0)
        with tempfile.TemporaryDirectory() as directory:
            path=os.path.join(directory,'search.json');graph.save(path);restored=SearchGraph.load(path)
        self.assertEqual(restored.missing('a'),[30.])
        restored.observe('a',30,'second.jpg',0)
        self.assertEqual(restored.next_action('a',self.tracked(restored,'a'),10.)['kind'],'map_frontier')

    def test_localization_recovery_keeps_coverage(self):
        graph=SearchGraph('robot');graph.place('a','room',headings=[0,30]);graph.observe('a',0,'image.jpg',0)
        self.assertEqual(graph.next_action('a',{},10.)['kind'],'relocalize')
        self.assertEqual(graph.next_action('a',self.tracked(graph,'a'),10.)['heading_degrees'],30.)
        self.assertEqual(graph.next_action('a',self.tracked(graph,'a'),11.)['kind'],'relocalize')

    def test_recognition_does_not_require_global_pose_or_claim_arrival(self):
        graph=SearchGraph('robot');graph.identify('robot.jpg',[565,178,639,238],'gpt-5.6-sol')
        action=graph.next_action('unknown',{},10.)
        self.assertEqual(action['kind'],'object_found')
        self.assertFalse(action['motion_authorized'])

    def test_route_is_rechecked_after_door_changes(self):
        graph=SearchGraph('robot');graph.place('a','a',headings=[]);graph.place('b','b')
        self.link(graph,'ab','a','b')
        action=graph.next_action('a',self.tracked(graph,'a'),10.)
        self.assertTrue(graph.validate_route(action,'a'))
        graph.block('ab')
        self.assertFalse(graph.validate_route(action,'a'))

    def test_deferred_frontier_does_not_starve_other_room(self):
        graph=SearchGraph('robot');graph.place('a','a',kind='frontier',headings=[]);graph.place('b','b')
        self.link(graph,'ab','a','b');graph.defer_frontier('a','door blocked')
        self.assertEqual(graph.next_action('a',self.tracked(graph,'a'),10.)['goal'],'b')
        graph.place('a','a',revision=1,kind='frontier',headings=[])
        self.assertEqual(graph.next_action('a',self.tracked(graph,'a'),10.)['kind'],'map_frontier')

    def test_new_target_reuses_map_but_not_coverage_or_old_reply(self):
        first=SearchGraph('tape');first.place('a','room',headings=[0]);first.observe('a',0,'old.jpg',0)
        ticket=first.recognition_ticket('old.jpg')
        second=first.new_search('robot')
        self.assertEqual(second.missing('a'),[0.])
        self.assertFalse(second.accept_recognition(ticket,[0,0,20,20],'sol'))
        self.assertIsNone(second.data['found'])

    def test_recognition_backlog_bounded_and_exact_image_bound(self):
        graph=SearchGraph('robot');a=graph.recognition_ticket('a.jpg');b=graph.recognition_ticket('b.jpg')
        with self.assertRaises(ValueError):graph.recognition_ticket('c.jpg')
        self.assertFalse(graph.accept_recognition(dict(a,image='different.jpg'),[0,0,20,20],'sol'))
        self.assertTrue(graph.accept_recognition(a,None,'sol'))
        self.assertTrue(graph.accept_recognition(b,[0,0,20,20],'sol'))
        self.assertEqual(graph.data['found']['image'],'b.jpg')
        self.assertFalse(graph.accept_recognition(b,[0,0,20,20],'sol'))

    def test_malformed_localization_time_fails_closed(self):
        graph=SearchGraph('robot');graph.place('a','room')
        for timestamp in (None,'10',True,float('nan'),float('inf'),11):
            pose=dict(self.tracked(graph,'a'),time=timestamp)
            self.assertEqual(graph.next_action('a',pose,10.)['kind'],'relocalize')


if __name__=='__main__':unittest.main()
