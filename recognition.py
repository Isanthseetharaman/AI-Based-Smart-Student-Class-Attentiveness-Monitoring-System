"""Local face recognition and observable geometry/object cues, without attention inference."""
from collections import Counter
import importlib
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
from config import (MODEL_DIR, FACE_MATCH_THRESHOLD, FACE_MATCH_MARGIN,
                    RECOGNITION_SECONDS, EAR_THRESHOLD, YAW_THRESHOLD,
                    PITCH_THRESHOLD, SIGNAL_DWELL_SECONDS, MIN_SIGNAL_CONFIDENCE,
                    CONVERSATION_DWELL_SECONDS, CONVERSATION_DISTANCE_RATIO,
                    PHONE_INFERENCE_INTERVAL, LIVE_MAX_DIMENSION,
                    PHONE_CONFIDENCE, PHONE_DWELL_SECONDS)


def choose_identity(scores, threshold=FACE_MATCH_THRESHOLD, margin=FACE_MATCH_MARGIN):
    """Reject low scores and ambiguous nearest neighbours; scores are not probabilities."""
    ranked = sorted(((key, float(score)) for key, score in scores if math.isfinite(score)), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] < threshold:
        return None
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < margin:
        return None
    return ranked[0]


class DwellTracker:
    def __init__(self, seconds, samples=3, max_gap=1.5):
        self.seconds, self.samples, self.max_gap = seconds, samples, max_gap
        self.history = {}

    def update(self, keys, now):
        keys = set(keys)
        self.history = {key: value for key, value in self.history.items() if key in keys}
        confirmed = set()
        for key in keys:
            first, last, count = self.history.get(key, (now, now, 0))
            if now - last > self.max_gap:
                first, count = now, 0
            count += 1
            self.history[key] = (first, now, count)
            if count >= self.samples and now - first >= self.seconds:
                confirmed.add(key)
        return confirmed


def eye_ratio(points, indices):
    p = points[list(indices), :2]
    width = np.linalg.norm(p[0] - p[3])
    if width < 1e-6:
        return None
    return float((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * width))


def geometry_cues(ear, yaw, pitch):
    if any(value is None or not math.isfinite(value) for value in (ear, yaw, pitch)):
        return []
    cues = []
    if ear < EAR_THRESHOLD:
        cues.append('Eyes closed')
    if abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD:
        cues.append('Looking away')
    return cues


class RecognitionEngine:
    def __init__(self, students):
        self.cv = importlib.import_module('cv2')
        self.detector = None
        self.recognizer = None
        self.gallery = {}
        self.messages = []
        yunet = MODEL_DIR / 'face_detection_yunet_2023mar.onnx'
        sface = MODEL_DIR / 'face_recognition_sface_2021dec.onnx'
        if not yunet.exists() or not sface.exists():
            self.messages.append('Recognition models missing. Run the model setup task.')
            return
        try:
            self.detector = self.cv.FaceDetectorYN.create(str(yunet), '', (320, 320), 0.9, 0.3, 5000)
            self.recognizer = self.cv.FaceRecognizerSF.create(str(sface), '')
            for student in students:
                if not student.get('face_image') or not student.get('consent_at'):
                    continue
                try:
                    content = np.fromfile(student['face_image'], dtype=np.uint8)
                    image = self.cv.imdecode(content, self.cv.IMREAD_COLOR)
                    if image is None:
                        raise ValueError('Unreadable image')
                    image = self.resize(image)
                    faces = self.detect(image)
                    if len(faces) != 1:
                        self.messages.append(f"{student['student_id']}: use a photo with exactly one clear face.")
                        continue
                    self.gallery[student['id']] = (student, self.feature(image, faces[0]))
                except Exception:
                    self.messages.append(f"{student['student_id']}: enrollment image could not be processed.")
        except Exception as exc:
            self.detector = None
            self.recognizer = None
            self.messages.append(f'Recognition could not start ({type(exc).__name__}); check OpenCV and model files.')
        self.messages.append(f'{len(self.gallery)} student face profiles loaded.')

    def resize(self, image, max_dimension=960):
        h, w = image.shape[:2]
        scale = min(1.0, max_dimension / max(h, w))
        return self.cv.resize(image, (round(w * scale), round(h * scale))) if scale < 1 else image

    def detect(self, image):
        if self.detector is None:
            return []
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        return [] if faces is None else faces

    def feature(self, image, face):
        aligned = self.recognizer.alignCrop(image, face)
        return self.recognizer.feature(aligned).copy()

    def recognize(self, image, face):
        if not self.gallery or min(face[2:4]) < 60:
            return None
        feature = self.feature(image, face)
        scores = [(pk, self.recognizer.match(feature, enrolled, self.cv.FaceRecognizerSF_FR_COSINE)) for pk, (_, enrolled) in self.gallery.items()]
        return choose_identity(scores)


class LandmarkCues:
    def __init__(self):
        self.landmarker = None
        self.status = 'Eye/head measurements unavailable: install MediaPipe and download models.'
        try:
            self.mp = importlib.import_module('mediapipe')
            # Import the Tasks API explicitly; some MediaPipe builds do not
            # expose the nested modules through ``mediapipe.tasks`` lazily.
            tasks = importlib.import_module('mediapipe.tasks.python')
            vision = tasks.vision
            model = MODEL_DIR / 'face_landmarker.task'
            if model.exists():
                options = vision.FaceLandmarkerOptions(
                    base_options=tasks.BaseOptions(model_asset_path=str(model)),
                    running_mode=vision.RunningMode.IMAGE, num_faces=1,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=True)
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
                self.status = 'Eye/head measurements ready'
        except Exception as exc:
            self.status += f' ({type(exc).__name__}: {exc})'

    def measure(self, image, face):
        if self.landmarker is None:
            return None
        h, w = image.shape[:2]
        x, y, fw, fh = face[:4]
        x1, y1 = max(0, int(x - fw * .2)), max(0, int(y - fh * .2))
        x2, y2 = min(w, int(x + fw * 1.2)), min(h, int(y + fh * 1.2))
        if x2 <= x1 or y2 <= y1:
            return None
        roi = image[y1:y2, x1:x2]
        rgb = np.ascontiguousarray(roi[:, :, ::-1])
        result = self.landmarker.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        if len(result.face_landmarks) != 1 or not result.facial_transformation_matrixes:
            return None
        points = np.array([(p.x * roi.shape[1], p.y * roi.shape[0], p.z) for p in result.face_landmarks[0]])
        left = eye_ratio(points, (33, 160, 158, 133, 153, 144))
        right = eye_ratio(points, (362, 385, 387, 263, 373, 380))
        if left is None or right is None:
            return None
        rotation = np.asarray(result.facial_transformation_matrixes[0])[:3, :3]
        # Remove uniform scale before extracting Euler angles.
        rotation = rotation / np.maximum(np.linalg.norm(rotation, axis=0), 1e-8)
        yaw = math.degrees(math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0])))
        pitch = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
        return (left + right) / 2, yaw, pitch

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()


class PhoneDetector:
    def __init__(self):
        self.model = None
        try:
            self.cv = importlib.import_module('cv2')
        except Exception:
            self.cv = None
        self.status = 'Phone detection unavailable: install Ultralytics and download models.'
        self._condition = threading.Condition()
        self._pending = None
        self._boxes = []
        self._closed = False
        self._worker = None
        path = MODEL_DIR / 'yolo11n.pt'
        try:
            if path.exists():
                os.environ.setdefault('YOLO_OFFLINE', 'true')
                os.environ.setdefault('YOLO_CONFIG_DIR', str(MODEL_DIR / 'settings'))
                ultralytics = importlib.import_module('ultralytics')
                ultralytics.settings.update({'sync': False})
                self.model = ultralytics.YOLO(str(path))
                self.status = 'Phone detection ready (fast background mode)'
                self._worker = threading.Thread(target=self._run, name='phone-detector', daemon=True)
                self._worker.start()
        except Exception as exc:
            self.status += f' ({type(exc).__name__}: {exc})'

    def detect(self, image):
        if self.model is None:
            return self._dark_phone_fallback(image)
        # Keep a large input size and a sensitive threshold: a phone's dark back
        # or a partially occluded phone is much harder to classify than a screen.
        # This runs on the background worker and therefore does not stall video frames.
        try:
            results = self.model.predict(image, classes=[67], conf=PHONE_CONFIDENCE, imgsz=768, device='cpu', verbose=False)
        except Exception:
            results = []
        boxes = [box.tolist() for result in results for box in result.boxes.xyxy.cpu().numpy()]
        return boxes or self._dark_phone_fallback(image)

    def _dark_phone_fallback(self, image):
        """Find a large dark phone back when the COCO model misses it."""
        if self.cv is None:
            return []
        height, width = image.shape[:2]
        gray = self.cv.cvtColor(image, self.cv.COLOR_BGR2GRAY)
        mask = self.cv.inRange(gray, 0, 62)
        mask = self.cv.morphologyEx(mask, self.cv.MORPH_CLOSE, self.cv.getStructuringElement(self.cv.MORPH_RECT, (9, 9)))
        contours, _ = self.cv.findContours(mask, self.cv.RETR_EXTERNAL, self.cv.CHAIN_APPROX_SIMPLE)
        candidates = []
        frame_area = float(width * height)
        for contour in contours:
            area = self.cv.contourArea(contour)
            x, y, w, h = self.cv.boundingRect(contour)
            if area < frame_area * 0.008 or area > frame_area * 0.55 or min(w, h) < 45:
                continue
            ratio = min(w, h) / max(w, h)
            rectangularity = area / max(1.0, w * h)
            rotated_w, rotated_h = self.cv.minAreaRect(contour)[1]
            rotated_ratio = min(rotated_w, rotated_h) / max(rotated_w, rotated_h, 1.0)
            # A hand can be attached to the phone, lowering contour
            # rectangularity; accept the rotated phone silhouette as well.
            if (ratio < 0.18 or ratio > 0.98) and (rotated_ratio < 0.18 or rotated_ratio > 0.98):
                continue
            if rectangularity < 0.20 and area < frame_area * 0.02:
                continue
            candidates.append((area, [float(x), float(y), float(x + w), float(y + h)]))
        return [box for _, box in sorted(candidates, reverse=True)[:1]]

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                image = self._pending
                self._pending = None
            try:
                boxes = self.detect(image)
            except Exception:
                boxes = []
            with self._condition:
                self._boxes = boxes

    def submit(self, image):
        """Queue only the newest frame; never block the live video callback."""
        if self.model is None:
            return
        with self._condition:
            self._pending = image.copy()
            self._condition.notify()

    def latest(self):
        with self._condition:
            return list(self._boxes)

    def close(self):
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)


class ClassroomPipeline:
    def __init__(self, students, enable_cues=True):
        self.faces = RecognitionEngine(students)
        self.landmarks = LandmarkCues() if enable_cues else None
        self.phones = PhoneDetector() if enable_cues else None
        self.last_phone_inference = 0.0
        self.phone_boxes = []
        self.phone_cues = DwellTracker(PHONE_DWELL_SECONDS, samples=1, max_gap=2.5)
        self.identities = DwellTracker(RECOGNITION_SECONDS)
        self.cues = DwellTracker(SIGNAL_DWELL_SECONDS)
        self.conversations = DwellTracker(CONVERSATION_DWELL_SECONDS)
        self.messages = list(self.faces.messages)
        if self.landmarks:
            self.messages.extend([self.landmarks.status, self.phones.status])

    @staticmethod
    def _phone_owner(phone_box, confirmed_faces):
        """Associate a phone only when it lies in one student's lower-body region."""
        px1, py1, px2, py2 = map(float, phone_box[:4])
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        candidates = []
        for student_pk, (fx, fy, fw, fh) in confirmed_faces.items():
            left, right = fx - fw * 0.8, fx + fw * 1.8
            top, bottom = fy - fh * 0.2, fy + fh * 3.5
            if left <= pcx <= right and top <= pcy <= bottom:
                distance = math.hypot(pcx - (fx + fw / 2), pcy - (fy + fh * 1.5))
                candidates.append((distance, student_pk))
        return min(candidates)[1] if candidates else None

    def process(self, image):
        image = self.faces.resize(image, LIVE_MAX_DIMENSION)
        matches, detected = [], self.faces.detect(image)
        # Run measurements on the original pixels, before drawing overlays.
        if self.phones and time.monotonic() - self.last_phone_inference >= PHONE_INFERENCE_INTERVAL:
            self.phones.submit(image)
            self.last_phone_inference = time.monotonic()
        if self.phones:
            self.phone_boxes = self.phones.latest()
        phones = self.phone_boxes if self.phones else []
        measurements_by_face = [self.landmarks.measure(image, face) if self.landmarks else None for face in detected]
        for face in detected:
            matches.append(self.faces.recognize(image, face))
        counts = Counter(match[0] for match in matches if match is not None)
        # Never assign one enrolled identity to two simultaneous faces.
        matches = [match if match is not None and counts[match[0]] == 1 else None for match in matches]
        now = time.monotonic()
        confirmed = self.identities.update([match[0] for match in matches if match], now)
        observations, visible_keys, display = [], [], []
        confirmed_faces = {}
        for index, (face, match) in enumerate(zip(detected, matches)):
            label, pk = 'Unknown', None
            score = None
            if match:
                pk, score = match
                student = self.faces.gallery[pk][0]
                label = f"{student['name']} ({score:.2f})"
                if pk not in confirmed:
                    label += ' confirming'
            measurements = measurements_by_face[index]
            cue_labels = geometry_cues(*measurements) if measurements else []
            # Unknown faces are visualized but never linked to a student record.
            if pk in confirmed:
                visible_keys.extend((pk, cue) for cue in cue_labels)
                confirmed_faces[pk] = tuple(map(float, face[:4]))
            display.append({'name': label, 'similarity': score, 'measurements': measurements})
            x, y, w, h = map(int, face[:4])
            self.faces.cv.rectangle(image, (x, y), (x + w, y + h), (40, 200, 80) if pk in confirmed else (0, 180, 255), 2)
            self.faces.cv.putText(image, label, (max(0, x), max(20, y - 8)), self.faces.cv.FONT_HERSHEY_SIMPLEX, .5, (40, 220, 80), 1)
            if measurements:
                ear, yaw, pitch = measurements
                text = f'EAR {ear:.2f} yaw {yaw:.0f} pitch {pitch:.0f}'
                self.faces.cv.putText(image, text, (max(0, x), min(image.shape[0]-5, y+h+18)), self.faces.cv.FONT_HERSHEY_SIMPLEX, .45, (255, 220, 0), 1)
            if cue_labels and pk in confirmed:
                self.faces.cv.putText(image, ' / '.join(cue_labels), (max(0, x), min(image.shape[0]-5, y+h+36)), self.faces.cv.FONT_HERSHEY_SIMPLEX, .52, (0, 180, 255), 2)
        phone_owners = [self._phone_owner(box, confirmed_faces) for box in phones]
        phone_keys = list(dict.fromkeys(owner if owner is not None else ('scene', index) for index, owner in enumerate(phone_owners)))
        phone_confirmed = self.phone_cues.update(phone_keys, now)
        for owner in phone_confirmed:
            # Phone detection is intentionally scene-level.  Even when a
            # phone is spatially close to a confirmed face, the camera cannot
            # reliably establish ownership or phone use by that student.
            observations.append({'student_pk': None, 'status': 'Phone visible'})
        for box in phones:
            x1, y1, x2, y2 = map(int, box)
            self.faces.cv.rectangle(image, (x1, y1), (x2, y2), (255, 140, 0), 2)
            owner = self._phone_owner(box, confirmed_faces)
            owner_name = self.faces.gallery[owner][0]['name'] if owner in self.faces.gallery else 'scene'
            self.faces.cv.putText(image, f'Using phone ({owner_name})', (x1, max(20, y1-8)), self.faces.cv.FONT_HERSHEY_SIMPLEX, .5, (255, 140, 0), 1)
        conversation_pairs = []
        confirmed_items = list(confirmed_faces.items())
        for index, (left_pk, left_box) in enumerate(confirmed_items):
            lx, ly, lw, lh = left_box
            lcx, lcy = lx + lw / 2, ly + lh / 2
            for right_pk, right_box in confirmed_items[index + 1:]:
                rx, ry, rw, rh = right_box
                rcx, rcy = rx + rw / 2, ry + rh / 2
                distance = math.hypot(lcx - rcx, lcy - rcy)
                limit = CONVERSATION_DISTANCE_RATIO * (min(lw, lh) + min(rw, rh)) / 2
                if distance <= limit:
                    conversation_pairs.append(tuple(sorted((left_pk, right_pk))))
        for pair in self.conversations.update(conversation_pairs, now):
            visible_keys.extend(((pair[0], 'Possible conversation'), (pair[1], 'Possible conversation')))
        sustained = self.cues.update(visible_keys, now)
        for pk, cue in sustained:
            observations.append({'student_pk': pk, 'status': cue})
        attendance = [{'student_pk': match[0], 'similarity': match[1]} for match in matches if match and match[0] in confirmed]
        return image, {'attendance': attendance, 'observations': observations,
                       'live_count': len(confirmed), 'detected_count': len(detected),
                       'faces': display, 'messages': self.messages, 'updated': time.monotonic()}

    def close(self):
        if self.landmarks:
            self.landmarks.close()
        if self.phones:
            self.phones.close()
