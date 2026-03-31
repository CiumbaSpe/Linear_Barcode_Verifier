from localizer import Localizer
from preproc import Preproc
from settings import *
from Verifier import Verifier

def main():

    preproc = Preproc(denoise=DENOISE, normalize_contrast=NORMALIZE_CONTRAST)
    localizer = Localizer()
    verifier = Verifier()

    gray, preprocessed = preproc.process(IMAGE_PATH)
    rect = localizer.localize(preprocessed)
    roi = verifier.verify_from_rect(rect, gray)


    print(f"Image: {IMAGE_PATH}")
    print(f"Shape: {gray.shape}")
    print(f"Denoise: {DENOISE}")
    print(f"Normalize contrast: {NORMALIZE_CONTRAST}")
    print(f"ROI shape: {roi.shape}")

    

if __name__ == "__main__":
    main()
