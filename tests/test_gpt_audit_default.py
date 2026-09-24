import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'local_nav'))
import gpt_audit


class AuditDefaultTests(unittest.TestCase):
    def test_missing_flag_enables_audit_when_run_pointer_exists(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, 'exists', return_value=True):
            self.assertTrue(gpt_audit.enabled())

    def test_explicit_false_disables_audit(self):
        for value in ('', '0', 'false', 'off'):
            with patch.dict(os.environ, {'JETBOT_GPT_AUDIT': value}, clear=True):
                self.assertFalse(gpt_audit.enabled())


if __name__ == '__main__':
    unittest.main()
