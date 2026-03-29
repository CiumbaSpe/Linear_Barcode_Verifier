from preproc import Preproc
from settings import *

def main():
    preproc = Preproc(
        denoise=DENOISE,
        normalize_contrast=NORMALIZE_CONTRAST,
    )

    gray, preprocessed = preproc.process(IMAGE_PATH)

    print(f"Image: {IMAGE_PATH}")
    print(f"Shape: {gray.shape}")
    print(f"Denoise: {DENOISE}")
    print(f"Normalize contrast: {NORMALIZE_CONTRAST}")

    if SAVE_OUTPUTS:
        gray_output = OUTPUT_DIR / f"{IMAGE_PATH.stem}_gray.png"
        preprocessed_output = OUTPUT_DIR / f"{IMAGE_PATH.stem}_preprocessed.png"

        preproc.save_image(gray, gray_output)
        preproc.save_image(preprocessed, preprocessed_output)

        print(f"Saved: {gray_output}")
        print(f"Saved: {preprocessed_output}")


if __name__ == "__main__":
    main()
