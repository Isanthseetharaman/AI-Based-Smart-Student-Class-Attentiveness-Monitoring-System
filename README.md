# Nexus Vision — classroom attendance and visible-cue review

Streamlit app with registered-photo face recognition, automatic attendance,
manual corrections, eye/head geometry measurements, scene-level phone detection,
and teacher review. It does not infer attention, understanding, emotion, or intent.

## Run

With a working Python 3.10+ installation:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_models.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

This workspace also has a project-local `.runtime` Python for the machine whose
original base Python was missing. It reads existing dependencies from `.venv` and
additional dependencies from `.runtime/Lib/site-packages`:

```powershell
.\.runtime\python.exe -m streamlit run app.py
```

Complete administrator setup locally before exposing the app to a network. Copy
`.env.example` to `.env` to customize settings; environment variables take priority.
Demo mode is off. Known shipped accounts using their unchanged demo passwords are
disabled outside demo mode, without deleting student or historical records.

## Use recognition

1. Register students and confirm consent in Students.
2. Upload a clear, front-facing photo containing exactly one face. Existing profiles
   have an Add / replace recognition photo form under Edit or delete a profile.
3. In Overview, select a specific class, enter the classroom, and enable automatic
   recording. Press START and allow the webcam. Stop/start after changing profiles.
4. The camera labels candidates with their name and cosine similarity. Unknown or
   ambiguous matches are not saved. A match needs at least three processed samples
   spanning two seconds. One identity cannot be assigned to two simultaneous faces.
5. Attendance is unique per student/date/room. Review and correct mistakes in
   Attendance. Removing an incorrect attendance record suppresses automatic re-entry
   for that student/date/room; manually marking present clears the suppression.

SFace similarity is not a probability. The default threshold is 0.50 with a 0.08
margin over the next candidate; validate these thresholds with the actual camera,
lighting and roster before relying on records. There is no liveness/anti-spoofing
check. Photographs can fool recognition. Teacher confirmation remains available.

## Visible cues

MediaPipe provides eye landmarks and a facial transformation matrix. The overlay
shows eye aspect ratio (EAR), yaw and pitch. Sustained low EAR records Eyes closed;
sustained yaw/pitch outside the thresholds records Looking away. These are geometric
measurements, not proof of sleepiness, distraction, or lack of understanding.
Missing landmarks produce no cue. Student-linked cues require a confirmed identity.

YOLO detects the COCO cell-phone class. Phone visible is a scene-level observation;
it is never assigned to an individual or described as phone usage. Cues require
sustained observations and have a 60-second per-cue cooldown before another record.
When two confirmed students remain close together for the configured dwell time, the
system records Possible conversation for both. This is a proximity cue and does not
prove that they are talking. Teachers can review each record and enter corrections in Analytics. Older historical
labels are preserved but the new pipeline only creates observable-cue labels.

All inference runs on the application server. No camera frames are uploaded to a
third-party API or stored by the pipeline. Registration photos remain under `data/`.
WebRTC transports the browser camera to the local/application server. Use localhost
or HTTPS with camera permission. Nonzero/RTSP sources currently provide snapshots
only; continuous recognition and dwell timing are available on the browser webcam.

## Models and dependencies

`setup_models.py` downloads official assets into ignored `models/` using temporary
files and size checks. Downloads occur only during setup, not on a camera frame.

- [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet): face detection.
- [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface): aligned face embeddings.
- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python): facial geometry; blendshape inference is disabled.
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/): cell-phone object boxes.

Follow each provider's model/software license, including Ultralytics' AGPL/enterprise
licensing terms. OpenCV main and contrib versions are aligned because MediaPipe and
Ultralytics declare different OpenCV distributions. Do not independently upgrade
one OpenCV wheel to version 5. CPU inference speed depends on hardware and class size;
there is no real-time performance guarantee. Missing modules/models are reported in
the camera panel. Models load once per stream, and inference is throttled.

## Storage, configuration and verification

SQLite, photos, and audit records live in `data/`. Back up together and restrict file
access. Establish retention and deletion procedures before deployment. Profiles with
history cannot be deleted through the UI. All dates use APP_TIMEZONE; historical class
filters use the current roster unless a camera event stored its class at capture.

Configuration includes FACE_MATCH_THRESHOLD, FACE_MATCH_MARGIN, RECOGNITION_SECONDS,
INFERENCE_INTERVAL, PHONE_INFERENCE_INTERVAL, LIVE_MAX_DIMENSION, EAR_THRESHOLD,
YAW_THRESHOLD, PITCH_THRESHOLD, SIGNAL_DWELL_SECONDS, MIN_SIGNAL_CONFIDENCE,
EVENT_COOLDOWN_SECONDS, CONVERSATION_DWELL_SECONDS and CONVERSATION_DISTANCE_RATIO.
The live pipeline drops frames while inference is busy, runs face/landmark processing
at the configured interval, and runs phone detection less often while reusing its
latest boxes. This keeps the preview current at the cost of occasional stale cue
overlays.

```powershell
.\.runtime\python.exe -m unittest -v test_workflows test_recognition
```

Tests use temporary databases and images. Camera model loading and a blank-frame
inference smoke check are separate from rule/database tests. An actual camera and
representative enrolled student images still require on-device validation.
