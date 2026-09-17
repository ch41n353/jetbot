"""Offline scorer checks. No network, model calls or robot access."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_nav'))
from benchmark_vlm_planning import SPATIAL, spatial_reference, score, canonical_answer


def box(a):
    return dict(zip(('x0','y0','x1','y1'),a))


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.case={'reference':dict(target_visible=True,target_box=[100,100,120,140],
                   contact_pixel=[110,140],obstacle_boxes=[[150,150,170,170],[180,180,200,200]])}
        self.answer=dict(target_visible=True,target_box=box([100,100,120,140]),
                         contact_pixel=dict(x=110,y=140),
                         obstacles=[dict(label='LEGO',box=box(a)) for a in self.case['reference']['obstacle_boxes']],
                         spatial_answers=[dict(id=t['id'],decision=spatial_reference(t)) for t in SPATIAL])

    def test_reference_and_correct_answer(self):
        self.assertEqual([spatial_reference(t) for t in SPATIAL],['execute','hold','hold','hold','execute'])
        self.assertTrue(score(self.answer,self.case,SPATIAL)['all_pass'])

    def test_unsafe_and_missing_spatial_answers(self):
        self.answer['spatial_answers'][1]['decision']='execute'
        self.answer['spatial_answers'].pop()
        s=score(self.answer,self.case,SPATIAL)
        self.assertEqual(s['unsafe_approvals'],1)
        self.assertEqual(s['spatial_correct'],3)
        self.assertFalse(s['all_pass'])

    def test_contact_error_and_obstacle_omission(self):
        self.answer['contact_pixel']['y']+=6
        self.answer['obstacles'].pop()
        s=score(self.answer,self.case,SPATIAL)
        self.assertFalse(s['target_pass'])
        self.assertEqual(s['obstacles_found'],1)

    def test_absent_target_cannot_have_hallucinated_box(self):
        self.case['reference']['target_visible']=False
        self.answer['target_visible']=False
        self.assertFalse(score(self.answer,self.case,SPATIAL)['target_pass'])
        self.answer.update(target_box=None,contact_pixel=None)
        self.assertTrue(score(self.answer,self.case,SPATIAL)['target_pass'])

    def test_touching_obstacle_is_blocked(self):
        self.assertEqual(spatial_reference(dict(free=[-50,-50,50,50],swept=[[0,0,10,10]],
                                               obstacles=[[10,0,20,10]])),'hold')

    def test_normalized_conversion_preserves_original(self):
        a=dict(target_box=box([250,250,500,500]),contact_pixel=dict(x=375,y=500),obstacles=[])
        b=canonical_answer(a,'normalized_1000')
        self.assertEqual(b['target_box'],box([160,120,320,240]))
        self.assertEqual(b['contact_pixel'],dict(x=240,y=240))
        self.assertEqual(a['target_box'],box([250,250,500,500]))
        a['contact_pixel']['x']=1001
        with self.assertRaises(ValueError):canonical_answer(a,'normalized_1000')


if __name__=='__main__':unittest.main()
