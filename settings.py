from pathlib import Path
import numpy as np


DATA_DIR = Path("codici-lineari-dati")
IMAGE_NAME = "UPC#03.BMP" # classic image
# IMAGE_NAME = "I25-LOW DECODABILITY IMGB.BMP" 
# IMAGE_NAME = "EAN-UPC-UPC-A MASTER GRADE IMGB.BMP" # vertical
# IMAGE_NAME = "EAN128-MASTER IMGB.BMP" # lot of writing
# IMAGE_NAME = "I25-CONTRAST IMGB.BMP" # boxed
# IMAGE_NAME = "ROTATED.BMP"

IMAGE_PATH = DATA_DIR / IMAGE_NAME

OUTPUT_DIR = Path("outputs")
SAVE_OUTPUTS = True

DENOISE = False
NORMALIZE_CONTRAST = False

# provare anche ps 24 e str 12
# 16, 8
PATCH_SIZE = 64
STRIDE = 32
EPS = 1e-6

ENERGY_TRAHSOLD = 50
COERENCE_TRASHOLD = 0.9
THETA_TRASHOLD = np.deg2rad(30)

SHRINK = 100
