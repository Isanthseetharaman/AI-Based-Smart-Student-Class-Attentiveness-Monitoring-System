from __future__ import annotations

from pathlib import Path
import hashlib
import sqlite3
import time

import pandas as pd
import streamlit as st

from config import APP_TITLE, CAMERA_SOURCE, DEMO_MODE, ensure_directories, local_now
import db
from reports import build_attendance_report
from uploads import save_face_image, remove_face_image
from vision import CameraProcessor, LiveVideoProcessor

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    from streamlit_webrtc.webrtc import SignallingTimeoutError
except ImportError:
    webrtc_streamer = None
    WebRtcMode = None
    SignallingTimeoutError = TimeoutError

st.set_page_config(page_title=APP_TITLE, page_icon='◉', layout='wide')
ensure_directories()

@st.cache_resource
def initialize_database():
    db.seed_demo_data()
    return True

initialize_database()
css = Path(__file__).parent / 'static' / 'styles.css'
if css.exists():
    st.markdown(f'<style>{css.read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)


def show_error(exc):
    if isinstance(exc, sqlite3.IntegrityError):
        st.error('That ID or username already exists, or the record has related history.')
    elif isinstance(exc, ValueError):
        st.error(str(exc))
    else:
        st.error('The operation could not be saved. Check file permissions and retry.')


def actor():
    return st.session_state.user['username']


def login():
    st.title('Smart Class Monitor')
    if db.needs_setup():
        st.info('Create the first administrator. Complete setup locally before exposing this app to a network.')
        with st.form('setup'):
            username = st.text_input('Administrator username')
            name = st.text_input('Full name')
            password = st.text_input('Password (at least 12 characters)', type='password')
            confirm = st.text_input('Confirm password', type='password')
            if st.form_submit_button('Create administrator'):
                try:
                    if password != confirm:
                        raise ValueError('Passwords do not match.')
                    db.create_user(username, name, password, 'admin')
                    st.rerun()
                except (ValueError, sqlite3.Error) as exc:
                    show_error(exc)
        return
    with st.form('login'):
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        if st.form_submit_button('Sign in'):
            user = db.authenticate(username, password)
            if user:
                st.session_state.user = user
                st.session_state.last_activity = time.time()
                st.rerun()
            st.error('Sign-in failed. After five failures, wait five minutes before retrying.')
    if DEMO_MODE:
        st.warning('Demo mode: admin / admin123 and teacher / teacher123. Use only demonstration data.')


def select_class(key, allow_all=True):
    classes = sorted({row['class_name'] for row in db.get_students()})
    options = ([None] if allow_all else []) + classes
    if not options:
        return None
    return st.selectbox('Class', options, format_func=lambda value: value or 'All classes', key=key)


@st.fragment(run_every=0.5)
def camera_status(context, class_name, room, auto_attendance, save_cues):
    if 'user' not in st.session_state or time.time() - st.session_state.get('last_activity', 0) > 1800:
        st.warning('Session expired. Sign in again to save camera results.')
        return
    processor = context.video_processor
    if not context.state.playing or processor is None:
        st.caption('Start the camera to recognize enrolled students.')
        return
    snapshot = processor.get_snapshot()
    live_count, detected_count = snapshot.get('live_count', 0), snapshot.get('detected_count', 0)
    st.metric('Live students detected', live_count, help='Students confirmed from consented enrollment photos after the confirmation dwell time.')
    st.caption(f'Faces in current frame: {detected_count} · Unknown or ambiguous faces are excluded from the student count.')
    for message in snapshot['messages']:
        st.caption(message)
    if snapshot['faces']:
        st.dataframe(pd.DataFrame(snapshot['faces']), hide_index=True, use_container_width=True)
    if time.monotonic() - snapshot['updated'] > 3:
        st.caption('Waiting for a fresh processed frame.')
        return
    if class_name and room.strip():
        if auto_attendance:
            for match in snapshot['attendance']:
                try:
                    db.record_camera_attendance(
                        match['student_pk'], room, class_name,
                        match['similarity'], actor=actor(),
                    )
                except (ValueError, sqlite3.Error) as exc:
                    show_error(exc)
        if save_cues:
            for cue in snapshot['observations']:
                try:
                    db.record_camera_observation(
                        cue['student_pk'], cue['status'], room, class_name,
                        actor=actor(),
                    )
                except (ValueError, sqlite3.Error) as exc:
                    show_error(exc)
    for cue in snapshot['observations']:
        st.info(cue['status'] + (' — scene-level' if cue['student_pk'] is None else ' — sustained visual cue'))
    stats = db.get_dashboard_stats(class_name)
    st.caption(f"Live attendance: {stats['present']} / {stats['students']} · Pending reviews: {stats['alerts']}")


def dashboard():
    st.title('Classroom overview')
    class_name = select_class('dashboard_class', allow_all=False)
    stats = db.get_dashboard_stats(class_name)
    for col, (label, value) in zip(st.columns(4), [
        ('Registered students', stats['students']), ('Present today', stats['present']),
        ('Not marked present', stats['students'] - stats['present']), ('Pending alerts (24h)', stats['alerts']),
    ]):
        col.metric(label, value)
    st.caption('Attendance uses the selected class roster and the configured local date. Unmarked students are not automatically confirmed absent.')
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader('Classroom camera')
        room = st.text_input('Camera classroom', value='Room 101')
        auto_attendance = st.checkbox('Automatically record confirmed matches', value=True)
        save_cues = st.checkbox('Measure visible cues and save for review', value=True)
        if class_name is None:
            st.info('Choose a specific class above to enable recognition and automatic records.')
        students = db.get_students(class_name) if class_name else []
        profile_key = hashlib.sha256(repr([(s['id'], s['face_image'], s['name'], s['consent_at']) for s in students]).encode()).hexdigest()[:12]
        if webrtc_streamer is not None and WebRtcMode is not None and str(CAMERA_SOURCE) == '0':
            # Use a named zero-argument factory: streamlit-webrtc invokes it when
            # the peer connection starts. The captured values are fixed for this
            # camera session and avoid constructor-argument ambiguity.
            def make_video_processor():
                return LiveVideoProcessor(students=students, enable_cues=save_cues)

            try:
                context = webrtc_streamer(key=f'classroom-camera-{class_name}-{room}-{profile_key}-{save_cues}', mode=WebRtcMode.SENDRECV,
                    video_processor_factory=make_video_processor,
                    media_stream_constraints={'video': True, 'audio': False}, async_processing=True)
                st.caption('Camera streaming' if context.state.playing else 'Camera stopped or awaiting permission')
                camera_status(context, class_name, room, auto_attendance, save_cues)
            except SignallingTimeoutError:
                st.error('Camera connection timed out. Refresh the page and allow camera access. Model loading now starts after connection; use HTTPS or localhost for browser camera access.')
        else:
            st.caption('Snapshot mode. Refresh explicitly to reconnect and capture a new frame.')
            if st.button('Capture / retry camera'):
                processor = CameraProcessor(CAMERA_SOURCE)
                frame = processor.read_frame()
                if frame is not None:
                    st.image(frame, channels='BGR', use_container_width=True)
                st.info(processor.status)
        st.caption('Registered-photo matches require sustained confirmation. Looking away and eye closure are visible cues, not attention scores. Phones are reported for the scene only. Possible conversation requires two confirmed students remaining close together; video cannot prove talking.')
        st.caption('Recognition has no liveness check. Review automatic attendance and correct errors in Attendance.')
    with right:
        st.subheader('Recorded observations (last 24 hours)')
        if stats['events']:
            st.bar_chart(pd.DataFrame(stats['events'])['status'].value_counts())
            st.caption('Counts are observations, not unique students or a live class distribution.')
        else:
            st.info('No observations recorded in the last 24 hours.')
        st.info('Teachers review observations before taking action. These signals do not establish a student’s understanding or intent.')


def students_page():
    st.title('Student management')
    if st.session_state.user['role'] == 'admin':
        with st.expander('Register student'):
            with st.form('student_form'):
                sid = st.text_input('Student ID', max_chars=200)
                name = st.text_input('Full name', max_chars=200)
                dept = st.text_input('Department', max_chars=200)
                cls = st.text_input('Class', max_chars=200)
                consent = st.checkbox('Consent and institutional authorization obtained')
                image = st.file_uploader('Optional face image: JPEG/PNG, maximum 5 MB', type=['jpg', 'jpeg', 'png'])
                if st.form_submit_button('Create profile'):
                    image_path = None
                    try:
                        if not all(value.strip() for value in (sid, name, dept, cls)) or not consent:
                            raise ValueError('Complete all fields and confirm consent.')
                        if image:
                            image_path = save_face_image(image.getvalue())
                        db.create_student(sid, name, dept, cls, image_path, actor=actor(), consent=consent)
                        st.success('Student registered.')
                    except (ValueError, sqlite3.Error, OSError) as exc:
                        remove_face_image(image_path)
                        show_error(exc)
    students = db.get_students()
    if not students:
        st.info('No students registered yet.')
        return
    search = st.text_input('Search students')
    visible = [row for row in students if search.lower() in f"{row['student_id']} {row['name']} {row['class_name']}".lower()]
    st.dataframe(pd.DataFrame(visible, columns=['student_id', 'name', 'department', 'class_name', 'created_at']), hide_index=True, use_container_width=True)
    if st.session_state.user['role'] == 'admin':
        with st.expander('Edit or delete a profile'):
            selected = st.selectbox('Student', students, format_func=lambda row: f"{row['student_id']} — {row['name']}")
            with st.form(f"photo_{selected['id']}"):
                photo = st.file_uploader('Add / replace recognition photo (one clear face)', type=['jpg', 'jpeg', 'png'])
                photo_consent = st.checkbox('Consent for student recognition confirmed')
                if st.form_submit_button('Save recognition photo'):
                    new_path = None
                    try:
                        if photo is None or not photo_consent:
                            raise ValueError('Upload a photo and confirm consent.')
                        new_path = save_face_image(photo.getvalue())
                        old_path = db.replace_student_image(selected['student_id'], new_path, actor=actor(), consent=photo_consent)
                        new_path = None  # The database now owns the saved image.
                        remove_face_image(old_path)
                        st.rerun()
                    except (ValueError, sqlite3.Error, OSError) as exc:
                        remove_face_image(new_path)
                        show_error(exc)
            with st.form(f"edit_{selected['id']}"):
                name = st.text_input('Name', selected['name'])
                department = st.text_input('Department', selected['department'])
                class_name = st.text_input('Class', selected['class_name'])
                if st.form_submit_button('Save changes'):
                    try:
                        db.update_student(selected['student_id'], actor=actor(), name=name, department=department, class_name=class_name)
                        st.rerun()
                    except (ValueError, sqlite3.Error) as exc:
                        show_error(exc)
            confirm = st.checkbox('Delete this profile and its image (only profiles without history)', key=f"delete_{selected['id']}")
            if st.button('Delete profile', disabled=not confirm):
                try:
                    image_path = db.delete_student(selected['student_id'], actor=actor())
                    remove_face_image(image_path)
                    st.rerun()
                except (ValueError, sqlite3.Error, OSError) as exc:
                    show_error(exc)


def attendance_page():
    st.title('Attendance')
    class_name = select_class('attendance_class')
    students = db.get_students(class_name)
    if students:
        with st.expander('Record or correct attendance', expanded=True):
            with st.form('attendance_entry'):
                student = st.selectbox('Student', students, format_func=lambda row: f"{row['student_id']} — {row['name']}")
                day = st.date_input('Attendance date', value=local_now().date(), max_value=local_now().date())
                room = st.text_input('Classroom', value='Room 101')
                present = st.selectbox('Action', ['Mark present', 'Remove incorrect attendance'])
                if st.form_submit_button('Save attendance'):
                    try:
                        db.set_attendance(student['id'], day, room, present == 'Mark present', actor=actor())
                        st.success('Attendance saved. Corrections are recorded in the audit log.')
                    except (ValueError, sqlite3.Error) as exc:
                        show_error(exc)
    a, b = st.columns(2)
    start = a.date_input('From', value=local_now().date().replace(day=1))
    end = b.date_input('To', value=local_now().date())
    if start > end:
        st.error('The start date must be on or before the end date.')
        return
    df = pd.DataFrame(db.get_attendance(start, end, class_name))
    if df.empty:
        st.info('No attendance found for this class and date range.')
        return
    st.download_button('Export CSV', build_attendance_report(df).to_csv(index=False), 'attendance_report.csv', 'text/csv')
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.bar_chart(df.groupby('attendance_date')['student_id'].nunique())
    st.caption('Daily chart counts distinct students across classrooms.')


def analytics_page():
    st.title('Observations and teacher review')
    class_name = select_class('analytics_class')
    students = db.get_students(class_name)
    st.caption('Teacher observations and camera cues are shown separately by source. Camera cues describe visible geometry or objects, not attention or understanding.')
    if students:
        with st.form('observation'):
            student = st.selectbox('Student', students, format_func=lambda row: f"{row['student_id']} — {row['name']}")
            status = st.selectbox('Observed signal', db.STATUSES)
            if st.form_submit_button('Record observation'):
                try:
                    db.record_attention(student['id'], status, actor=actor())
                    st.success('Observation recorded for review.')
                except (ValueError, sqlite3.Error) as exc:
                    show_error(exc)
    events = db.get_attention_events(class_name)
    if not events:
        st.info('No observations recorded.')
        return
    st.subheader('Recorded observation history')
    st.bar_chart(pd.DataFrame(events)['status'].value_counts())
    st.dataframe(pd.DataFrame(events)[['student_code', 'name', 'status', 'source', 'event_time', 'reviewed', 'review_note', 'reviewer']], hide_index=True)
    pending = [event for event in events if not event['reviewed']]
    if pending:
        with st.form('review'):
            event = st.selectbox('Pending review', pending, format_func=lambda row: f"#{row['id']} · {row['name']} · {row['status']} · {row['event_time']}")
            note = st.text_area('Review decision / correction')
            if st.form_submit_button('Complete review'):
                try:
                    db.review_attention(event['id'], note, actor=actor())
                    st.rerun()
                except (ValueError, sqlite3.Error) as exc:
                    show_error(exc)


def accounts_page():
    st.title('Create staff account')
    with st.form('staff_account'):
        username = st.text_input('Username')
        name = st.text_input('Full name')
        password = st.text_input('Initial password (at least 12 characters)', type='password')
        role = st.selectbox('Role', ['teacher', 'admin'])
        if st.form_submit_button('Create account'):
            try:
                db.create_user(username, name, password, role, actor=actor())
                st.success('Account created.')
            except (ValueError, sqlite3.Error) as exc:
                show_error(exc)


def main():
    if 'user' in st.session_state:
        user = db.get_user(st.session_state.user['username'])
        if not user or time.time() - st.session_state.get('last_activity', 0) > 1800:
            st.session_state.clear()
        else:
            st.session_state.user = user
            st.session_state.last_activity = time.time()
    if 'user' not in st.session_state:
        login()
        return
    pages = {'Overview': dashboard, 'Students': students_page, 'Attendance': attendance_page, 'Analytics': analytics_page}
    if st.session_state.user['role'] == 'admin':
        pages['Accounts'] = accounts_page
    with st.sidebar:
        st.title('NEXUS VISION')
        st.write(st.session_state.user['name'])
        st.caption(st.session_state.user['role'].upper())
        page = st.radio('Navigation', list(pages))
        if DEMO_MODE:
            st.warning('Demo mode')
        if st.button('Sign out'):
            st.session_state.clear()
            st.rerun()
    pages[page]()


if __name__ == '__main__':
    main()
