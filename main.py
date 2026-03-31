import csv
from localizer import Localizer
from preproc import Preproc
from settings import *
from Verifier import Verifier

def single_test():
    preproc = Preproc(denoise=DENOISE, normalize_contrast=NORMALIZE_CONTRAST)
    localizer = Localizer()
    verifier = Verifier()

    gray, preprocessed = preproc.process(IMAGE_PATH)
    rect = localizer.localize(preprocessed)
    contrast, modulation = verifier.verify_from_rect(rect, gray)


    print(f"Image: {IMAGE_PATH}")
    print(f"Shape: {gray.shape}")
    print(f"Denoise: {DENOISE}")
    print(f"Normalize contrast: {NORMALIZE_CONTRAST}")
    print(f"Contrast: {contrast}")
    print(f"Modulation: {modulation}")

def full_test():
    preproc = Preproc(denoise=DENOISE, normalize_contrast=NORMALIZE_CONTRAST)
    localizer = Localizer()
    verifier = Verifier()

    rows = []
    for image_path in sorted(DATA_DIR.glob("*.BMP")):
        gray, preprocessed = preproc.process(image_path)
        rect = localizer.localize(preprocessed)
        contrast, modulation = verifier.verify_from_rect(rect, gray)
        rows.append([image_path.name, contrast, modulation])

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "contrast", "modulation"])
        writer.writerows(rows)

def main():
    single_test()
    # full_test()

if __name__ == "__main__":
    main()
