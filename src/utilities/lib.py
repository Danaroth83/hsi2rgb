from enum import Enum

import numpy as np


Row, Col, Chn = int, int, int
ImageArray = np.ndarray[tuple[Row, Col, Chn], np.float32]


class ImageFormatEnum(str, Enum):
    """Available list of implemented image formats."""
    ENVI = "envi"
    MAT_FILE = "mat_file"
    SKIMAGE = "skimage"
    NUMPY = "numpy"
