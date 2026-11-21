from io import BytesIO
from pathlib import Path
from uuid import uuid4
import warnings

from PIL import Image, UnidentifiedImageError
from config import DATA_DIR

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def save_face_image(content):
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError('Choose an image smaller than 5 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                if image.format not in ('JPEG', 'PNG') or image.width * image.height > 16_000_000:
                    raise ValueError('Choose a JPEG or PNG with at most 16 million pixels.')
                image.load()
                normalized = image.convert('RGB')
                normalized.thumbnail((1600, 1600))
                directory = (DATA_DIR / 'faces').resolve()
                directory.mkdir(parents=True, exist_ok=True)
                destination = directory / f'{uuid4().hex}.jpg'
                normalized.save(destination, format='JPEG', quality=90)
                return str(destination)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('The upload is not a valid supported image.') from exc


def remove_face_image(filename):
    if filename:
        path = Path(filename).resolve()
        if path.parent == (DATA_DIR / 'faces').resolve():
            path.unlink(missing_ok=True)
