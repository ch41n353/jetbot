import os,sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'local_nav'))
from fit_floor_alignment import validate_config,collect_pairs,ROOT

class FloorAlignmentTests(unittest.TestCase):
    def test_invalid_scan_bounds_and_missing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output=os.path.join(os.path.dirname(directory),'candidate.json')
            for limit,step in [(9,.25),(0,.25),(float('nan'),.25),(8,.1),(1,2),(8,float('inf'))]:
                with self.subTest(limit=limit,step=step), self.assertRaises(ValueError):
                    validate_config(directory,output,limit,step)
            with self.assertRaises(ValueError):validate_config(directory+'/missing',output,8,.25)
    def test_forbids_calibration_and_observation_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            for output in [os.path.join(ROOT,'calibration/imu_mount.json'),os.path.join(directory,'record.json')]:
                with self.assertRaises(ValueError):validate_config(directory,output,8,.25)
            validate_config(directory,os.path.join(os.path.dirname(directory),'candidate.json'),8,.25)
    def test_insufficient_data_does_not_create_tracker(self):
        with tempfile.TemporaryDirectory() as directory, patch('fit_floor_alignment.FloorTracker') as tracker:
            with self.assertRaisesRegex(ValueError,'Insufficient recorded frames'):
                collect_pairs(directory,{}, {}, {})
            tracker.assert_not_called()
    def test_excessive_record_count_is_bounded(self):
        with patch('fit_floor_alignment.glob.glob',return_value=['x']*1001):
            with self.assertRaisesRegex(ValueError,'at most 1000'):
                collect_pairs('/unused',{}, {}, {})

if __name__=='__main__':unittest.main()
