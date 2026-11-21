from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
import hashlib
import hmac
import secrets
import sqlite3
import time

from config import DB_PATH, DEMO_MODE, local_now

STATUSES = ('Eyes closed', 'Looking away', 'Possible conversation', 'Phone visible', 'Unknown')

@contextmanager
def connect():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA foreign_keys=ON')
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    return salt, hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 120_000).hex()


def init_db():
    with connect() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, name TEXT, role TEXT, salt TEXT, password_hash TEXT);
        CREATE TABLE IF NOT EXISTS students(id INTEGER PRIMARY KEY, student_id TEXT UNIQUE, name TEXT, department TEXT, class_name TEXT, face_image TEXT, consent_at TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS attendance(id INTEGER PRIMARY KEY, student_id INTEGER, attendance_date TEXT, attendance_time TEXT, classroom TEXT, UNIQUE(student_id, attendance_date, classroom), FOREIGN KEY(student_id) REFERENCES students(id));
        CREATE TABLE IF NOT EXISTS attention_events(id INTEGER PRIMARY KEY, student_id INTEGER, status TEXT, confidence REAL, event_time TEXT, reviewed INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS login_attempts(username TEXT PRIMARY KEY, failures INTEGER, blocked_until REAL);
        CREATE TABLE IF NOT EXISTS audit_log(id INTEGER PRIMARY KEY, actor TEXT, action TEXT, detail TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS attendance_overrides(student_id INTEGER, attendance_date TEXT, classroom TEXT, PRIMARY KEY(student_id,attendance_date,classroom), FOREIGN KEY(student_id) REFERENCES students(id));
        CREATE INDEX IF NOT EXISTS attendance_date_idx ON attendance(attendance_date);
        CREATE INDEX IF NOT EXISTS event_time_idx ON attention_events(event_time);
        ''')
        columns = {row['name'] for row in c.execute('PRAGMA table_info(attention_events)')}
        for name, declaration in [('source', "TEXT DEFAULT 'legacy'"), ('review_note', "TEXT DEFAULT ''"), ('reviewer', 'TEXT'), ('reviewed_at', 'TEXT'), ('classroom', 'TEXT'), ('class_name', 'TEXT')]:
            if name not in columns:
                c.execute(f'ALTER TABLE attention_events ADD COLUMN {name} {declaration}')
        if not DEMO_MODE:
            # Disable only the known shipped passwords, preserving customized accounts.
            for username, password in [('admin', 'admin123'), ('teacher', 'teacher123')]:
                row = c.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
                if row and hmac.compare_digest(_hash(password, row['salt'])[1], row['password_hash']):
                    c.execute('DELETE FROM users WHERE id=?', (row['id'],))


def _audit(c, actor, action, detail):
    c.execute('INSERT INTO audit_log(actor,action,detail,created_at) VALUES(?,?,?,?)',
              (actor, action, detail, local_now().isoformat()))


def needs_setup():
    with connect() as c:
        return not c.execute("SELECT 1 FROM users WHERE role='admin'").fetchone()


def create_user(username, name, password, role='teacher', actor=None):
    username, name = username.strip(), name.strip()
    if not username or not name or len(password) < 12 or role not in ('admin', 'teacher'):
        raise ValueError('Enter a username, name and password of at least 12 characters.')
    with connect() as c:
        c.execute('BEGIN IMMEDIATE')
        existing_admin = c.execute("SELECT 1 FROM users WHERE role='admin'").fetchone()
        if existing_admin:
            _authorize(c, actor, admin=True)
        elif role != 'admin':
            raise ValueError('Create the administrator first.')
        salt, hashed = _hash(password)
        c.execute('INSERT INTO users(username,name,role,salt,password_hash) VALUES(?,?,?,?,?)', (username, name, role, salt, hashed))
        _audit(c, actor or username, 'create_user', username)


def seed_demo_data():
    init_db()
    if DEMO_MODE:
        with connect() as c:
            if not c.execute('SELECT 1 FROM users').fetchone():
                for username, password, role in [('admin', 'admin123', 'admin'), ('teacher', 'teacher123', 'teacher')]:
                    salt, hashed = _hash(password)
                    c.execute('INSERT INTO users(username,name,role,salt,password_hash) VALUES(?,?,?,?,?)', (username, username.title(), role, salt, hashed))


def authenticate(username, password):
    username = username.strip()
    with connect() as c:
        c.execute('BEGIN IMMEDIATE')
        attempt = c.execute('SELECT * FROM login_attempts WHERE username=?', (username,)).fetchone()
        if attempt and attempt['blocked_until'] > time.time():
            return None
        row = c.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
        hashed = _hash(password, row['salt'] if row else 'missing-user')[1]
        if row and hmac.compare_digest(hashed, row['password_hash']):
            c.execute('DELETE FROM login_attempts WHERE username=?', (username,))
            return {key: row[key] for key in ('id', 'username', 'name', 'role')}
        failures = (attempt['failures'] if attempt and not attempt['blocked_until'] else 0) + 1
        c.execute('INSERT OR REPLACE INTO login_attempts VALUES(?,?,?)', (username, failures, time.time() + 300 if failures >= 5 else 0))
        return None


def get_user(username):
    with connect() as c:
        row = c.execute('SELECT id,username,name,role FROM users WHERE username=?', (username,)).fetchone()
        return dict(row) if row else None


def _authorize(c, actor, admin=False):
    row = c.execute('SELECT role FROM users WHERE username=?', (actor,)).fetchone()
    if not row or row['role'] not in (('admin',) if admin else ('admin', 'teacher')):
        raise ValueError('Your account is not authorized for this action.')


def create_student(student_id, name, department, class_name, face_image=None, *, actor, consent=False):
    values = [value.strip() for value in (student_id, name, department, class_name)]
    if not all(values) or not consent or any(len(value) > 200 for value in values):
        raise ValueError('All profile fields and consent are required (maximum 200 characters each).')
    with connect() as c:
        _authorize(c, actor, admin=True)
        c.execute('INSERT INTO students(student_id,name,department,class_name,face_image,consent_at,created_at) VALUES(?,?,?,?,?,?,?)', (*values, face_image, local_now().isoformat(), local_now().isoformat()))
        _audit(c, actor, 'create_student', values[0])


def update_student(student_id, *, actor, **fields):
    allowed = {key: str(value).strip() for key, value in fields.items() if key in {'name', 'department', 'class_name'}}
    if not allowed or not all(allowed.values()):
        raise ValueError('Profile fields cannot be empty.')
    with connect() as c:
        _authorize(c, actor, admin=True)
        c.execute(f"UPDATE students SET {','.join(key+'=?' for key in allowed)} WHERE student_id=?", (*allowed.values(), student_id))
        _audit(c, actor, 'update_student', student_id)


def delete_student(student_id, *, actor):
    # Preserve historical records: profiles with history cannot be deleted.
    with connect() as c:
        _authorize(c, actor, admin=True)
        row = c.execute('SELECT id,face_image FROM students WHERE student_id=?', (student_id,)).fetchone()
        if not row:
            raise ValueError('Student not found.')
        if c.execute('SELECT 1 FROM attendance WHERE student_id=? UNION ALL SELECT 1 FROM attention_events WHERE student_id=?', (row['id'], row['id'])).fetchone():
            raise ValueError('This student has attendance or review history and cannot be deleted.')
        c.execute('DELETE FROM attendance_overrides WHERE student_id=?', (row['id'],))
        c.execute('DELETE FROM students WHERE id=?', (row['id'],))
        _audit(c, actor, 'delete_student', student_id)
        return row['face_image']


def get_students(class_name=None):
    with connect() as c:
        return [dict(row) for row in c.execute('SELECT * FROM students' + (' WHERE class_name=?' if class_name else '') + ' ORDER BY name', (class_name,) if class_name else ())]


def set_attendance(student_pk, day, classroom, present, *, actor):
    if not classroom.strip() or day > local_now().date():
        raise ValueError('Enter a classroom and a date that is not in the future.')
    with connect() as c:
        _authorize(c, actor)
        if not c.execute('SELECT 1 FROM students WHERE id=?', (student_pk,)).fetchone():
            raise ValueError('Student not found.')
        if present:
            c.execute('DELETE FROM attendance_overrides WHERE student_id=? AND attendance_date=? AND classroom=?', (student_pk, day.isoformat(), classroom.strip()))
            c.execute('INSERT OR IGNORE INTO attendance(student_id,attendance_date,attendance_time,classroom) VALUES(?,?,?,?)', (student_pk, day.isoformat(), local_now().strftime('%H:%M:%S'), classroom.strip()))
        else:
            c.execute('DELETE FROM attendance WHERE student_id=? AND attendance_date=? AND classroom=?', (student_pk, day.isoformat(), classroom.strip()))
            c.execute('INSERT OR IGNORE INTO attendance_overrides VALUES(?,?,?)', (student_pk, day.isoformat(), classroom.strip()))
        _audit(c, actor, 'attendance_correction', f'{student_pk}: {day} / {classroom.strip()} / present={present}')


def get_attendance(start=None, end=None, class_name=None):
    query = 'SELECT s.student_id,s.name,s.class_name,a.attendance_date,a.attendance_time,a.classroom FROM attendance a JOIN students s ON s.id=a.student_id WHERE 1=1'
    args = []
    for condition, value in [('a.attendance_date>=?', start), ('a.attendance_date<=?', end), ('s.class_name=?', class_name)]:
        if value is not None:
            query += ' AND ' + condition
            args.append(str(value))
    with connect() as c:
        return [dict(row) for row in c.execute(query + ' ORDER BY a.attendance_date DESC,a.attendance_time DESC', args)]


def record_attention(student_pk, status, *, actor):
    if status not in STATUSES:
        raise ValueError('Unknown status.')
    with connect() as c:
        _authorize(c, actor)
        if not c.execute('SELECT 1 FROM students WHERE id=?', (student_pk,)).fetchone():
            raise ValueError('Student not found.')
        c.execute("INSERT INTO attention_events(student_id,status,confidence,event_time,source) VALUES(?,?,NULL,?,'teacher observation')", (student_pk, status, local_now().isoformat()))
        _audit(c, actor, 'record_observation', f'{student_pk}: {status}')


def get_attention_events(class_name=None):
    with connect() as c:
        query = "SELECT e.*,COALESCE(s.name,'Camera scene') AS name,COALESCE(s.student_id,'—') AS student_code FROM attention_events e LEFT JOIN students s ON s.id=e.student_id"
        return [dict(row) for row in c.execute(query + (' WHERE COALESCE(e.class_name,s.class_name)=?' if class_name else '') + ' ORDER BY e.event_time DESC', (class_name,) if class_name else ())]


def review_attention(event_id, note, *, actor):
    if not note.strip():
        raise ValueError('Enter a review decision or correction.')
    with connect() as c:
        _authorize(c, actor)
        result = c.execute('UPDATE attention_events SET reviewed=1,review_note=?,reviewer=?,reviewed_at=? WHERE id=? AND reviewed=0', (note.strip(), actor, local_now().isoformat(), event_id))
        if result.rowcount != 1:
            raise ValueError('This event has already been reviewed or no longer exists.')
        _audit(c, actor, 'review_attention', str(event_id))


def get_dashboard_stats(class_name=None):
    students = get_students(class_name)
    attendance = get_attendance(local_now().date(), local_now().date(), class_name)
    cutoff = (local_now() - timedelta(hours=24)).isoformat()
    events = [event for event in get_attention_events(class_name) if event['event_time'] >= cutoff]
    return {'students': len(students), 'present': len({row['student_id'] for row in attendance}),
            'alerts': sum(not event['reviewed'] and event['status'] not in ('Attentive', 'Unknown') for event in events),
            'events': events}


def record_camera_attendance(student_pk, classroom, class_name, similarity, *, actor):
    from config import FACE_MATCH_THRESHOLD
    if not classroom.strip() or not class_name or not FACE_MATCH_THRESHOLD <= similarity <= 1.00001:
        raise ValueError('A confirmed face match, class and classroom are required.')
    now = local_now()
    with connect() as c:
        _authorize(c, actor)
        student = c.execute('SELECT * FROM students WHERE id=? AND class_name=?', (student_pk, class_name)).fetchone()
        if not student or not student['consent_at'] or not student['face_image']:
            raise ValueError('Student is no longer enrolled for recognition in this class.')
        if c.execute('SELECT 1 FROM attendance_overrides WHERE student_id=? AND attendance_date=? AND classroom=?', (student_pk, now.date().isoformat(), classroom.strip())).fetchone():
            return False
        result = c.execute('INSERT OR IGNORE INTO attendance(student_id,attendance_date,attendance_time,classroom) VALUES(?,?,?,?)', (student_pk, now.date().isoformat(), now.strftime('%H:%M:%S'), classroom.strip()))
        if result.rowcount:
            _audit(c, actor, 'camera_attendance', f'{student_pk}: {classroom.strip()} / similarity={similarity:.3f}')
        return bool(result.rowcount)


def record_camera_observation(student_pk, status, classroom, class_name, *, actor):
    from config import EVENT_COOLDOWN_SECONDS
    if status not in ('Eyes closed', 'Looking away', 'Possible conversation', 'Phone visible') or not classroom.strip() or not class_name:
        raise ValueError('A visible cue, class and classroom are required.')
    if status == 'Phone visible' and student_pk is not None:
        raise ValueError('Phone observations are scene-level and cannot be assigned to a student.')
    now = local_now()
    cutoff = (now - timedelta(seconds=EVENT_COOLDOWN_SECONDS)).isoformat()
    with connect() as c:
        c.execute('BEGIN IMMEDIATE')
        _authorize(c, actor)
        if student_pk is not None:
            student = c.execute('SELECT consent_at FROM students WHERE id=? AND class_name=?', (student_pk, class_name)).fetchone()
            if not student or not student['consent_at']:
                raise ValueError('Student is no longer enrolled in this class.')
        if c.execute("SELECT 1 FROM attention_events WHERE student_id IS ? AND status=? AND classroom=? AND class_name=? AND source='camera cue' AND event_time>=?", (student_pk, status, classroom.strip(), class_name, cutoff)).fetchone():
            return False
        c.execute("INSERT INTO attention_events(student_id,status,event_time,source,classroom,class_name) VALUES(?,?,?,'camera cue',?,?)", (student_pk, status, now.isoformat(), classroom.strip(), class_name))
        _audit(c, actor, 'camera_observation', f'{student_pk}: {status} / {classroom.strip()}')
        return True


def replace_student_image(student_id, face_image, *, actor, consent):
    if not consent or not face_image:
        raise ValueError('Confirm consent and upload a face image.')
    with connect() as c:
        _authorize(c, actor, admin=True)
        row = c.execute('SELECT face_image FROM students WHERE student_id=?', (student_id,)).fetchone()
        if row is None:
            raise ValueError('Student not found.')
        c.execute('UPDATE students SET face_image=?,consent_at=? WHERE student_id=?', (face_image, local_now().isoformat(), student_id))
        _audit(c, actor, 'replace_face_image', student_id)
        return row['face_image']
