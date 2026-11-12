"""The availability pages map every statement item and state fulfilment only when every item is fulfilled."""
import copy
import unittest
from scripts import availability, statement_check
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
WORDS = ("projected", "awaiting", "not yet", "CPU")
class ReviewPackagePresentation(unittest.TestCase):
    def test_statement_page_maps_every_item(self):
        evaluation = statement_check.evaluate(ROOT)
        before = copy.deepcopy(evaluation)
        text = statement_check.render(evaluation, demo=statement_check.demonstration(evaluation, ROOT))
        for row in evaluation['items']:
            self.assertIn(f"| {row['phrase']} | {statement_check.PROVIDED[row['id']]} |", text)
        for word in WORDS:
            self.assertNotIn(word, text)
        self.assertEqual(evaluation, before)
    def test_fulfilment_is_stated_only_when_every_item_is_fulfilled(self):
        evaluation = statement_check.evaluate(ROOT)
        complete = evaluation['counts'][statement_check.FULFILLED] == len(evaluation['items'])
        self.assertEqual('Every item is fulfilled.' in statement_check.render(evaluation), complete)
        done = copy.deepcopy(evaluation)
        for row in done['items']:
            row['status'] = statement_check.FULFILLED
        done['counts'] = {statement_check.FULFILLED: len(done['items']), statement_check.AWAITING: 0,
                          statement_check.MISSING: 0}
        self.assertIn('Every item is fulfilled.', statement_check.render(done))
    def test_availability_page_points_to_the_manifest_and_the_command(self):
        text = availability.render_repository_release(availability.state(ROOT))
        self.assertIn('release/paper_artifacts.json', text)
        self.assertIn('make paper', text)
        for word in WORDS:
            self.assertNotIn(word, text)
    def test_current_clone_commands_and_scope_note(self):
        text = (ROOT/'README.md').read_text()
        self.assertIn('git clone https://github.com/VKHS/CROSS-MORSE.git', text)
        self.assertNotIn('github.com/VKHS/CR-9', text)
        self.assertLess(text.index('## Package scope'), text.index('git clone '))
        self.assertIn('badge.svg?branch=main', text)
