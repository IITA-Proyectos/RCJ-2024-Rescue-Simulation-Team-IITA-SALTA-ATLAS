import numpy as np
from fixture_detection.color_filter import ColorFilter

class NonFixtureFilter:
    def __init__(self) -> None:
        self.color_filters = (
            ColorFilter((94, 112,  32), (95, 143, 139)), # walls
            ColorFilter((0, 0, 192,), (0, 0, 192)), # Normal Floor
            ColorFilter((0, 0, 29), (0, 0, 138)), # Obstacles   anteriores valores que funcionan:((0, 0, 30), (0, 0, 138))
            ColorFilter((0, 0, 10), (0, 0, 30)), # Outer Hole old value ((0, 0, 8), (0, 2, 30)),
            ColorFilter((0, 0, 106), (0, 0, 106)), # Inner Hole old value ((0, 0, 105), (0, 0, 108)),
            ColorFilter((0, 205, 233), (0, 205, 233)), # Red Tile
            ColorFilter((107, 0, 72), (116, 90, 211)) # Checkpoint

        )

    def filter(self, image):
        base = np.zeros(image.shape[:2], np.bool_)
        for f in self.color_filters:
            filtered = f.filter(image).astype(np.bool_)
            base += filtered

        return base
    