import csv
from Grader import Grader
from Localizer import Localizer
from Preproc import Preproc
from settings import *
from Verifier import Verifier

def single_test():
    preproc = Preproc(denoise=DENOISE, normalize_contrast=NORMALIZE_CONTRAST)
    localizer = Localizer()
    verifier = Verifier()
    grader = Grader()

    gray, preprocessed = preproc.process(IMAGE_PATH)
    rect = localizer.localize(preprocessed)
    verification = verifier.verify_from_rect(rect, gray)
    metrics = verification["mean"]
    grade = grader.grade_symbol(verification["scanlines"])


    print(f"Image: {IMAGE_PATH}")
    print(f"Shape: {gray.shape}")
    print(f"Denoise: {DENOISE}")
    print(f"Normalize contrast: {NORMALIZE_CONTRAST}")
    print(f"Min reflectance: {metrics['min_reflectance']}")
    print(f"Min edge contrast: {metrics['min_edge_contrast']}")
    print(f"Contrast: {metrics['contrast']}")
    print(f"Modulation: {metrics['modulation']}")
    print(f"Defects: {metrics['defects']}")
    print(f"Grade: {grade['grade']}")
    print(f"Numeric grade: {grade['numeric_grade']}")

def full_test():
    preproc = Preproc(denoise=DENOISE, normalize_contrast=NORMALIZE_CONTRAST)
    localizer = Localizer()
    verifier = Verifier()
    grader = Grader()

    rows = []
    for image_path in sorted(DATA_DIR.glob("*.BMP")):
        gray, preprocessed = preproc.process(image_path)
        rect = localizer.localize(preprocessed)
        verification = verifier.verify_from_rect(rect, gray)
        metrics = verification["mean"]
        grade = grader.grade_symbol(verification["scanlines"])
        rows.append([
            image_path.name,
            metrics["min_reflectance"],
            metrics["min_edge_contrast"],
            metrics["contrast"],
            metrics["modulation"],
            metrics["defects"],
            grade["numeric_grade"],
            grade["grade"],
        ])

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "min_reflectance", "min_edge_contrast", "contrast", "modulation", "defects", "numeric_grade", "grade"])
        writer.writerows(rows)

def main():
    single_test()
    # full_test()

if __name__ == "__main__":
    main()
