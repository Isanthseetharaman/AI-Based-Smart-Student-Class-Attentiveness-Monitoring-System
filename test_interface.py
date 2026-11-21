import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
from types import SimpleNamespace

import db
from streamlit.testing.v1 import AppTest


class InterfaceTests(unittest.TestCase):
    def test_authenticated_pages_render(self):
        # AppTest has no WebRTC session manager; mock only the browser component.
        camera = SimpleNamespace(state=SimpleNamespace(playing=False), video_processor=None)
        component = patch('streamlit_webrtc.webrtc_streamer', return_value=camera)
        component.start()
        self.addCleanup(component.stop)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(db, 'DB_PATH', Path(directory) / 'ui.db'), patch.object(db, 'DEMO_MODE', False):
                db.init_db()
                db.create_user('owner', 'Owner', 'a-strong-password', 'admin')
                db.create_student('S1', 'Student', 'Science', 'Class A', actor='owner', consent=True)
                app = AppTest.from_file('app.py', default_timeout=30)
                app.run()
                self.assertFalse(app.exception)
                app.text_input[0].set_value('owner')
                app.text_input[1].set_value('a-strong-password')
                app.button[0].click().run()
                self.assertFalse(app.exception)
                for page in ['Students', 'Attendance', 'Analytics', 'Accounts', 'Overview']:
                    app.sidebar.radio[0].set_value(page).run()
                    self.assertFalse(app.exception, str(app.exception))


if __name__ == '__main__':
    unittest.main()
