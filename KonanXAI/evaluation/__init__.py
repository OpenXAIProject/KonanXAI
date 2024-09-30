from .pixel_flipping import (
    PixelFlipping,
    MoRF,
    LeRF,
    AbPC,
)


PIXEL_FLIPPING_METRICS = [
    MoRF,
    LeRF,
    AbPC,
]

AVAILABLE_METRICS = PIXEL_FLIPPING_METRICS
