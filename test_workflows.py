import unittest
from datetime import timedelta
from io import BytesIO
from pathlib import Path
import sqlite3
import tempfile
from unittest.mock import patch

from PIL import Image
import db
import uploads
from config import local_now
from vision import classify_attention, CameraProcessor


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [patch.object(db, 'DB_PATH', self.root / 'test.db'), patch.object(db, 'DEMO_MODE', False), patch.object(uploads, 'DATA_DIR', self.root)]
        for item in self.patches:
            item.start()
        db.init_db()
        db.create_user('owner', 'Owner', 'a-strong-password', 'admin')
        db.create_user('staff', 'Teacher', 'another-long-password', actor='owner')
        db.create_student('S1', 'Student One', 'Science', 'Class A', actor='owner', consent=True)
        self.student = db.get_students()[0]

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_authentication_and_lockout(self):
        user = db.authenticate('owner', 'a-strong-password')
        self.assertEqual(set(user), {'id', 'username', 'name', 'role'})
        for _ in range(5):
            self.assertIsNone(db.authenticate('owner', 'wrong'))
        self.assertIsNone(db.authenticate('owner', 'a-strong-password'))
        with patch.object(db.time, 'time', return_value=db.time.time() + 301):
            self.assertIsNotNone(db.authenticate('owner', 'a-strong-password'))

    def test_unique_attendance_and_correction(self):
        for room in ('A', 'A', 'B'):
            db.set_attendance(self.student['id'], local_now().date(), room, True, actor='staff')
        self.assertEqual(len(db.get_attendance()), 2)
        self.assertEqual(db.get_dashboard_stats('Class A')['present'], 1)
        self.assertEqual(db.get_dashboard_stats('Class B')['present'], 0)
        db.set_attendance(self.student['id'], local_now().date(), 'A', False, actor='staff')
        self.assertEqual(len(db.get_attendance()), 1)
        with self.assertRaises(ValueError):
            db.delete_student('S1', actor='owner')
        with self.assertRaises(ValueError):
            db.set_attendance(self.student['id'], local_now().date() + timedelta(days=1), 'A', True, actor='staff')

    def test_authorization_and_foreign_keys(self):
        with self.assertRaises(ValueError):
            db.create_student('S2', 'Student', 'Science', 'Class A', actor='staff', consent=True)
        with self.assertRaises(sqlite3.IntegrityError):
            with db.connect() as c:
                c.execute("INSERT INTO attendance(student_id) VALUES(99999)")
        with self.assertRaises(sqlite3.IntegrityError):
            db.create_student('S1', 'Duplicate', 'Science', 'Class A', actor='owner', consent=True)

    def test_review_and_audit(self):
        db.record_attention(self.student['id'], 'Looking away', actor='staff')
        self.assertEqual(db.get_dashboard_stats()['alerts'], 1)
        event = db.get_attention_events()[0]
        db.review_attention(event['id'], 'Looking at the classroom board; no action.', actor='staff')
        self.assertEqual(db.get_dashboard_stats()['alerts'], 0)
        self.assertEqual(db.get_attention_events()[0]['reviewer'], 'staff')
        with self.assertRaises(ValueError):
            db.review_attention(event['id'], 'Overwrite', actor='staff')
        with db.connect() as c:
            self.assertTrue(c.execute("SELECT 1 FROM audit_log WHERE action='review_attention'").fetchone())

    def test_upload_validation_and_cleanup(self):
        with self.assertRaises(ValueError):
            uploads.save_face_image(b'not an image')
        with self.assertRaises(ValueError):
            uploads.save_face_image(b'x' * (uploads.MAX_UPLOAD_BYTES + 1))
        content = BytesIO()
        Image.new('RGB', (20, 20)).save(content, 'PNG')
        filename = uploads.save_face_image(content.getvalue())
        self.assertEqual(Path(filename).parent, self.root / 'faces')
        self.assertTrue(Path(filename).exists())
        uploads.remove_face_image(filename)
        self.assertFalse(Path(filename).exists())
        outside = self.root / 'keep.txt'
        outside.write_text('keep')
        uploads.remove_face_image(str(outside))
        self.assertTrue(outside.exists())

    def test_sustained_signals(self):
        self.assertEqual(classify_attention(.1, 0, 0, closed_seconds=.2), 'Unknown')
        self.assertEqual(classify_attention(.1, 0, 0, closed_seconds=2.1), 'Eyes closed')
        self.assertEqual(classify_attention(None, 0, 0), 'Unknown')
        self.assertEqual(classify_attention(.3, 35, 0, distracted_seconds=3), 'Looking away')
        self.assertEqual(classify_attention(.3, 0, 0, phone=True, phone_seconds=.1), 'Unknown')
        self.assertEqual(classify_attention(.3, 0, 0), 'No sustained cue')

    def test_migration_disables_only_default_passwords(self):
        with db.connect() as c:
            salt, hashed = db._hash('admin123')
            c.execute('INSERT INTO users(username,name,role,salt,password_hash) VALUES(?,?,?,?,?)', ('admin', 'Demo', 'admin', salt, hashed))
        db.init_db()
        self.assertIsNone(db.get_user('admin'))
        self.assertIsNotNone(db.get_user('owner'))
        db.init_db()
        self.assertEqual(len(db.get_students()), 1)

    def test_capture_released_after_failure(self):
        from unittest.mock import MagicMock
        fake = MagicMock()
        fake.VideoCapture.return_value.read.return_value = (False, None)
        with patch('vision.cv2', fake):
            self.assertIsNone(CameraProcessor('rtsp://example.invalid').read_frame())
        fake.VideoCapture.return_value.release.assert_called_once()

    def test_incomplete_opencv_does_not_crash_video_processor(self):
        from types import SimpleNamespace
        from vision import LiveVideoProcessor
        with patch('vision.cv2', SimpleNamespace()):
            processor = LiveVideoProcessor()
            self.assertIsNone(processor.detector.cascade)
            self.assertEqual(processor.detector.detect(object()), [])
            self.assertIn('incomplete', processor.detector.status)

    def test_missing_cascade_does_not_crash(self):
        from unittest.mock import MagicMock
        from vision import FaceDetector
        fake = MagicMock()
        fake.data.haarcascades = 'missing-cascades'
        fake.CascadeClassifier.return_value.empty.return_value = True
        with patch('vision.cv2', fake):
            detector = FaceDetector()
            self.assertIsNone(detector.cascade)
            self.assertEqual(detector.detect(object()), [])


if __name__ == '__main__':
    unittest.main()
