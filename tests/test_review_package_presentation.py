"""Presentation must not upgrade CPU availability to original-paper completion."""
import copy
import unittest
from scripts import availability, statement_check
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
class ReviewPackagePresentation(unittest.TestCase):
    def test_package_note_does_not_change_registration(self):
        package = {'release':'https://example.invalid/release','published':31,'artifacts':37}
        before = copy.deepcopy(package)
        text = availability.published_package_note(package)
        self.assertIn('31 of 37', text)
        self.assertIn('distinct', text)
        self.assertEqual(package, before)
    def test_absent_package_is_not_claimed_published(self):
        self.assertIn('not registered', availability.published_package_note(None))
    def test_availability_preserves_the_evaluated_count(self):
        state = availability.state(ROOT)
        evaluation = state['release']['compliance']
        text = availability.render_repository_release(state)
        expected = str(evaluation['counts']['fulfilled']) + ' of ' + str(len(evaluation['items']))
        self.assertIn(expected, text)
        self.assertIn('original-paper channel', text)
        self.assertIn('Published execution package', text)
    def test_statement_presentation_preserves_the_evaluation(self):
        evaluation = statement_check.evaluate(ROOT)
        before = copy.deepcopy(evaluation)
        demo = statement_check.demonstration(evaluation, ROOT)
        text = statement_check.render(evaluation, demo=demo)
        expected = str(evaluation['counts']['fulfilled']) + ' of ' + str(len(evaluation['items']))
        self.assertIn(expected, text)
        self.assertEqual(evaluation, before)
    def test_current_clone_commands_and_scope_note(self):
        text = (ROOT/'README.md').read_text()
        self.assertIn('git clone https://github.com/VKHS/CROSS-MORSE.git', text)
        self.assertNotIn('github.com/VKHS/CR-9', text)
        self.assertLess(text.index('## Package scope'), text.index('git clone '))
        self.assertIn('badge.svg?branch=main', text)
