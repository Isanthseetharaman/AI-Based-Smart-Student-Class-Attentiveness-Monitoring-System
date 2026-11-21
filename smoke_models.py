"""Verify model loading and inference without using student images or a camera."""
import numpy as np
from recognition import ClassroomPipeline

pipeline = ClassroomPipeline([], enable_cues=True)
try:
    print('\n'.join(pipeline.messages), flush=True)
    assert pipeline.faces.detector is not None, 'YuNet did not load'
    assert pipeline.faces.recognizer is not None, 'SFace did not load'
    assert pipeline.landmarks.landmarker is not None, 'MediaPipe did not load'
    assert pipeline.phones.model is not None, 'YOLO did not load'
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    output, snapshot = pipeline.process(blank)
    assert output.shape == blank.shape
    assert not snapshot['attendance'] and not snapshot['observations']
    # Exercise the landmark model even though YuNet correctly finds no face.
    assert pipeline.landmarks.measure(blank, (100, 100, 150, 150)) is None
    print('All models loaded; blank-frame inference produced no records.', flush=True)
finally:
    pipeline.close()
