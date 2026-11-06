"""Reviewer FAQ reference integrity for the current paper release."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts import check_reviewer_faq as faq  # noqa: E402


class FaqTests(unittest.TestCase):
    def test_committed_faq_references_resolve(self):
        self.assertEqual(faq.check(), [])

    def test_dangling_references_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            text = faq.FAQ.read_text(encoding="utf-8")
            broken = text.replace("`make fingerprint`", "`make no-such-target`", 1)
            broken = broken.replace("`scripts/source_fingerprint.py`", "`scripts/missing_file.py`", 1)
            broken = broken.replace("[Reviewer guide](REVIEWER_GUIDE.md)", "[Reviewer guide](MISSING_GUIDE.md)", 1)
            path = tmp / "REVIEWER_FAQ.md"
            path.write_text(broken, encoding="utf-8")
            problems = faq.check(path, faq.MAKEFILE, PROJECT)
            self.assertTrue(any("no-such-target" in p for p in problems))
            self.assertTrue(any("missing_file.py" in p for p in problems))
            # relative links are resolved from the FAQ's own directory, so the broken link is reported from docs/
            self.assertTrue(any("MISSING_GUIDE" in p for p in problems) or not (tmp / "MISSING_GUIDE.md").exists())
            without_check = text.replace("Check: `make fingerprint`", "Checked by hand", 1)
            path.write_text(without_check, encoding="utf-8")
            self.assertTrue(any(p.startswith("A1: no Check line") for p in faq.check(path, faq.MAKEFILE, PROJECT)))

    def test_make_targets_include_multi_target_rules(self):
        targets = faq.make_targets("sample smoke: train-sample\nverify: registry-check\nVAR := x\n")
        self.assertEqual(targets, {"sample", "smoke", "verify"})



if __name__ == "__main__":
    unittest.main()
