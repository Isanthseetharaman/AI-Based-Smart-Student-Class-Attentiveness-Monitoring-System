import pandas as pd

def build_attendance_report(df):
    if df is None or df.empty: return pd.DataFrame(columns=["student_id","name","attendance_date","attendance_time","classroom"])
    report = df.copy()
    # Prevent user-controlled cells being interpreted as spreadsheet formulas.
    for column in report.select_dtypes(include=['object', 'string']).columns:
        report[column] = report[column].map(
            lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@', '\t', '\r', '\n')) else value
        )
    return report

def build_attention_report():
    return pd.DataFrame({"Signal": ["Head pose", "Eye closure", "Phone object", "Face identity"], "Meaning": ["Yaw/pitch outside configured range", "EAR below threshold for sustained interval", "YOLO class confidence above threshold", "Known or unknown face match"], "Review": ["Teacher review recommended", "Teacher review recommended", "Teacher review recommended", "Registration consent required"]})
