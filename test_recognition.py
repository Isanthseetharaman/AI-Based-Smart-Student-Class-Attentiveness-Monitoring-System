import unittest
from datetime import timedelta
from unittest.mock import patch

import db
from config import local_now
from recognition import DwellTracker, choose_identity, geometry_cues
import test_workflows


class RecognitionRulesTests(unittest.TestCase):
    def test_reject_unknown_and_ambiguous_faces(self):
        self.assertIsNone(choose_identity([]))
        self.assertIsNone(choose_identity([(1, .3)]))
        self.assertIsNone(choose_identity([(1, .8), (2, .77)]))
        self.assertEqual(choose_identity([(1, .8), (2, .5)]), (1, .8))
        self.assertIsNone(choose_identity([(1, float('nan'))]))

    def test_confirmation_requires_time_and_consecutive_samples(self):
        tracker = DwellTracker(seconds=2, samples=3)
        self.assertEqual(tracker.update([1], 0), set())
        self.assertEqual(tracker.update([1], 1), set())
        self.assertEqual(tracker.update([1], 2), {1})
        self.assertEqual(tracker.update([], 2.1), set())
        self.assertEqual(tracker.update([1], 2.2), set())

    def test_camera_gap_resets_confirmation(self):
        tracker = DwellTracker(seconds=2)
        tracker.update([1], 0)
        tracker.update([1], 1)
        self.assertEqual(tracker.update([1], 8), set())
        self.assertEqual(tracker.update([1], 9), set())
        self.assertEqual(tracker.update([1], 10), {1})

    def test_cues_describe_only_visible_measurements(self):
        self.assertEqual(geometry_cues(.1, 0, 0), ['Eyes closed'])
        self.assertEqual(geometry_cues(.3, 40, 0), ['Looking away'])
        self.assertEqual(geometry_cues(.3, 0, 0), [])
        self.assertEqual(geometry_cues(None, 0, 0), [])


class CameraDatabaseTests(unittest.TestCase):
    def setUp(self):
        test_workflows.WorkflowTests.setUp(self)
        db.replace_student_image('S1', 'test-profile.jpg', actor='owner', consent=True)

    tearDown = test_workflows.WorkflowTests.tearDown

    def test_auto_attendance_preserves_manual_override(self):
        pk = self.student['id']
        self.assertTrue(db.record_camera_attendance(pk, 'A', 'Class A', .8, actor='staff'))
        self.assertFalse(db.record_camera_attendance(pk, 'A', 'Class A', .8, actor='staff'))
        db.set_attendance(pk, local_now().date(), 'A', False, actor='staff')
        self.assertFalse(db.record_camera_attendance(pk, 'A', 'Class A', .8, actor='staff'))
        self.assertEqual(db.get_attendance(), [])
        db.set_attendance(pk, local_now().date(), 'A', True, actor='staff')
        self.assertEqual(len(db.get_attendance()), 1)

    def test_auto_attendance_checks_class_and_score(self):
        with self.assertRaises(ValueError):
            db.record_camera_attendance(self.student['id'], 'A', 'Class B', .9, actor='staff')
        with self.assertRaises(ValueError):
            db.record_camera_attendance(self.student['id'], 'A', 'Class A', .1, actor='staff')

    def test_camera_cues_are_deduplicated_and_reviewable(self):
        pk = self.student['id']
        self.assertTrue(db.record_camera_observation(pk, 'Eyes closed', 'A', 'Class A', actor='staff'))
        self.assertFalse(db.record_camera_observation(pk, 'Eyes closed', 'A', 'Class A', actor='staff'))
        self.assertTrue(db.record_camera_observation(None, 'Phone visible', 'A', 'Class A', actor='staff'))
        events = db.get_attention_events('Class A')
        self.assertEqual(len(events), 2)
        phone = next(row for row in events if row['status'] == 'Phone visible')
        self.assertIsNone(phone['student_id'])
        self.assertEqual(phone['name'], 'Camera scene')
        db.review_attention(phone['id'], 'Phone was on the teacher desk.', actor='staff')
        with self.assertRaises(ValueError):
            db.record_camera_observation(pk, 'Phone visible', 'A', 'Class A', actor='staff')
        with self.assertRaises(ValueError):
            db.record_camera_observation(pk, 'Distracted', 'A', 'Class A', actor='staff')


if __name__ == '__main__':
    unittest.main()

