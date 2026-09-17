import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_nav'))
from compare_vlm_cost import estimate_cost


class CostTests(unittest.TestCase):
    def test_cached_writes_and_reasoning_not_double_counted(self):
        usage=dict(input_tokens=1000,output_tokens=100,
            input_tokens_details=dict(cached_tokens=200,cache_write_tokens=100),
            output_tokens_details=dict(reasoning_tokens=90))
        self.assertAlmostEqual(estimate_cost(usage,[4,.4,20,5]),.00538)

    def test_uncached(self):
        self.assertAlmostEqual(estimate_cost(dict(input_tokens=2000,output_tokens=500),[5,.5,30,None]),.025)

    def test_invalid_and_unknown_write_rate(self):
        with self.assertRaises(ValueError):
            estimate_cost(dict(input_tokens=10,output_tokens=1,input_tokens_details=dict(cached_tokens=20)),[5,.5,30,None])
        with self.assertRaises(ValueError):
            estimate_cost(dict(input_tokens=10,output_tokens=1,input_tokens_details=dict(cache_write_tokens=2)),[5,.5,30,None])


if __name__=='__main__':unittest.main()
