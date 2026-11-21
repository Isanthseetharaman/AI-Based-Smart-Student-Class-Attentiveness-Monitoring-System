from __future__ import annotations
from pathlib import Path
import math
import importlib
import threading
import time
from config import INFERENCE_INTERVAL
from config import EAR_THRESHOLD, YAW_THRESHOLD, PITCH_THRESHOLD, SIGNAL_DWELL_SECONDS, MIN_SIGNAL_CONFIDENCE

try:
    cv2 = importlib.import_module('cv2')
except (ImportError, OSError):
    cv2 = None

try:
    VideoProcessorBase = importlib.import_module('streamlit_webrtc').VideoProcessorBase
except ImportError:
    class VideoProcessorBase:
        pass

class CameraProcessor:
    """One bounded snapshot; retry creates a fresh capture and always releases it."""
    def __init__(self, source):
        self.source = int(source) if str(source).isdigit() else source
        self.status = 'Not connected'
        self.capture = None

    def read_frame(self):
        if cv2 is None:
            self.status = 'OpenCV is unavailable'
            return None
        try:
            if isinstance(self.source, str):
                self.capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
                ])
            else:
                self.capture = cv2.VideoCapture(self.source)
            ok, frame = self.capture.read()
            self.status = 'Snapshot received' if ok else 'Camera unavailable; retry to reconnect'
            return frame if ok else None
        except Exception:
            self.status = 'Camera connection failed; check the source configuration'
            return None
        finally:
            if self.capture is not None:
                self.capture.release()
                self.capture = None


class FaceDetector:
    def __init__(self):
        self.cascade = None
        self.status = 'Face detection unavailable: OpenCV is missing or incomplete.'
        if str(getattr(cv2, '__version__', '')).split('.')[0] == '5':
            self.status = 'Face detection requires OpenCV 4.x. Install opencv-python>=4.8,<5 and restart the app.'
            return
        required = ('CascadeClassifier', 'cvtColor', 'COLOR_BGR2GRAY',
                    'rectangle', 'putText', 'FONT_HERSHEY_SIMPLEX')
        if cv2 is None or any(not hasattr(cv2, name) for name in required):
            return
        cascade_directory = getattr(getattr(cv2, 'data', None), 'haarcascades', None)
        if not cascade_directory:
            self.status = 'Face detection unavailable: OpenCV cascade data is missing.'
            return
        try:
            cascade = cv2.CascadeClassifier(str(Path(cascade_directory) / 'haarcascade_frontalface_default.xml'))
            if cascade.empty():
                self.status = 'Face detection unavailable: the face cascade could not be loaded.'
                return
            self.cascade = cascade
            self.status = 'Face detection ready'
        except Exception:
            self.status = 'Face detection unavailable: OpenCV could not initialize the detector.'

    def detect(self, frame):
        if cv2 is None or frame is None or self.cascade is None or self.cascade.empty():
            return []
        return self.cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)


def classify_attention(ear, yaw, pitch, phone=False, closed_seconds=0, distracted_seconds=0, phone_seconds=0, confidence=1.0):
    """Classify validated measurements only; missing or transient signals are unknown."""
    values = (ear, yaw, pitch, closed_seconds, distracted_seconds, phone_seconds, confidence)
    if any(value is None or not math.isfinite(value) for value in values):
        return 'Unknown'
    if not 0 <= confidence <= 1 or confidence < MIN_SIGNAL_CONFIDENCE or not 0 <= ear <= 1 or min(closed_seconds, distracted_seconds, phone_seconds) < 0:
        return 'Unknown'
    if phone:
        return 'Phone visible' if phone_seconds >= SIGNAL_DWELL_SECONDS else 'Unknown'
    if ear < EAR_THRESHOLD:
        return 'Eyes closed' if closed_seconds >= SIGNAL_DWELL_SECONDS else 'Unknown'
    if abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD:
        return 'Looking away' if distracted_seconds >= SIGNAL_DWELL_SECONDS else 'Unknown'
    return 'No sustained cue'


class LiveVideoProcessor(VideoProcessorBase):
    def __init__(self, students=None, enable_cues=False):
        self.detector = FaceDetector()
        self.students = students
        self.enable_cues = enable_cues
        self.pipeline_initialized = False
        self.pipeline = None
        self.lock = threading.RLock()
        self.last_inference = 0.0
        self.last_image = None
        self.snapshot = {'attendance': [], 'observations': [], 'faces': [], 'live_count': 0, 'detected_count': 0, 'messages': [], 'updated': 0}
        if students is not None:
            self.snapshot['messages'] = ['Camera connected. Loading recognition models…']

    def get_snapshot(self):
        with self.lock:
            return dict(self.snapshot)

    def on_ended(self):
        with self.lock:
            if self.pipeline:
                self.pipeline.close()
                self.pipeline = None

    def recv(self, frame):
        image = frame.to_ndarray(format='bgr24')
        # Drop frames while inference is busy so latency does not accumulate.
        if not self.lock.acquire(blocking=False):
            from av import VideoFrame
            return VideoFrame.from_ndarray(image, format='bgr24')
        try:
            if self.students is not None and not self.pipeline_initialized:
                # Model construction is intentionally deferred until after the
                # WebRTC worker has answered the browser offer. Loading YOLO,
                # MediaPipe and face models can exceed WebRTC's 10-second offer timeout.
                self.pipeline_initialized = True
                try:
                    from recognition import ClassroomPipeline
                    self.pipeline = ClassroomPipeline(self.students, self.enable_cues)
                    self.snapshot['messages'] = self.pipeline.messages
                except Exception as exc:
                    self.snapshot['messages'] = [f'Recognition unavailable ({type(exc).__name__}). Check dependencies and model setup.']
            if self.pipeline:
                if time.monotonic() - self.last_inference >= INFERENCE_INTERVAL:
                    try:
                        image, self.snapshot = self.pipeline.process(image)
                        self.last_image = image
                    except Exception as exc:
                        self.pipeline.identities.history.clear()
                        self.pipeline.cues.history.clear()
                        self.snapshot = {'attendance': [], 'observations': [], 'faces': [], 'live_count': 0, 'detected_count': 0,
                                         'messages': [f'Frame processing failed ({type(exc).__name__}); no records saved.'], 'updated': 0}
                        self.last_image = None
                    self.last_inference = time.monotonic()
                elif self.last_image is not None:
                    image = self.last_image
            elif cv2 is not None:
                for x, y, w, h in self.detector.detect(image):
                    cv2.rectangle(image, (x, y), (x + w, y + h), (69, 53, 220), 2)
                    cv2.putText(image, 'Face detected', (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (69, 53, 220), 2)
            from av import VideoFrame
            return VideoFrame.from_ndarray(image, format='bgr24')
        finally:
            self.lock.release()
