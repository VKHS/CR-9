"""Local badge generation preserves the registered message and avoids external requests."""
import unittest
import xml.etree.ElementTree as ET
from scripts.render_evidence_badge import render_badge

class EvidenceBadgeTests(unittest.TestCase):
    def test_registered_message_is_preserved(self):
        value={'schemaVersion':1,'label':'availability statement','message':'artifact publication workflow','color':'blue'}
        text=render_badge(value)
        root=ET.fromstring(text)
        self.assertEqual(root.attrib['aria-label'],'availability statement: artifact publication workflow')
        self.assertNotIn('verified',text)
        self.assertNotIn('raw.githubusercontent.com',text)
        self.assertNotIn('shields.io',text)
    def test_xml_text_is_escaped(self):
        value={'schemaVersion':1,'label':'A&B','message':'<unverified>','color':'blue'}
        root=ET.fromstring(render_badge(value))
        self.assertEqual(root.attrib['aria-label'],'A&B: <unverified>')
    def test_invalid_color_is_rejected(self):
        with self.assertRaises(ValueError):
            render_badge({'schemaVersion':1,'label':'a','message':'b','color':'url(evil)'})
