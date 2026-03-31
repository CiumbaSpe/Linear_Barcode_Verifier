from localizer import Localizer
from preproc import Preproc
from settings import *

def main():
    preproc = Preproc(
        denoise=DENOISE,
        normalize_contrast=NORMALIZE_CONTRAST,
    )

    gray, preprocessed = preproc.process(IMAGE_PATH)
    localizer = Localizer()
    rect = localizer.localize(preprocessed)


    print(f"Image: {IMAGE_PATH}")
    print(f"Shape: {gray.shape}")
    print(f"Denoise: {DENOISE}")
    print(f"Normalize contrast: {NORMALIZE_CONTRAST}")

    

if __name__ == "__main__":
    main()
