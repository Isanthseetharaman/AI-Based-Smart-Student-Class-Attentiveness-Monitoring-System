"""Download only official model assets; run once before starting the camera."""
from pathlib import Path
import urllib.request

MODEL_DIR = Path(__file__).resolve().parent / 'models'
MODELS = {
    'face_detection_yunet_2023mar.onnx': ('https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx', 200_000),
    'face_recognition_sface_2021dec.onnx': ('https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx', 30_000_000),
    'face_landmarker.task': ('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 1_000_000),
    'yolo11n.pt': ('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt', 1_000_000),
}


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    for filename, (url, minimum_size) in MODELS.items():
        destination = MODEL_DIR / filename
        if destination.exists() and destination.stat().st_size >= minimum_size:
            print(f'Already available: {filename}', flush=True)
            continue
        temporary = destination.with_suffix(destination.suffix + '.download')
        print(f'Downloading {filename}', flush=True)
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'NexusVision-ModelSetup/1.0'})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if temporary.stat().st_size < minimum_size:
                raise ValueError(f'{filename}: incomplete download')
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    print('All four models are available.', flush=True)


if __name__ == '__main__':
    main()
