from .api import *
import cv2

class Image:
    def __init__(self, w: int, h: int, c: int, raw: bytes):
        self.struct: IMAGE = make_image(w, h, c)
        self._memset(raw)

    def _memset(self, byte_data: bytes):
        copy_image_from_bytes(self.struct, byte_data)
        
    def __del__(self):
        self.free()
        
    def free(self):
        free_image(self.struct)
        
    def shape(self):
        pass

    def data(self) -> IMAGE:
        return self.struct

def open_image(img_path: str, size: tuple=None, channel: str='RGB') -> Image:
    raw = cv2.imread(img_path)
    if channel in ('GRAY', 'GREY'):
        pass
    elif channel in ('RGB'):
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    if size is not None:
        raw = cv2.resize(raw, size)
    h, w, c = raw.shape
    img = Image(w, h, c, raw.tobytes())
    return img
