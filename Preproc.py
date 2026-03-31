from pathlib import Path

import cv2


class Preproc:
    def __init__(self, denoise=False, normalize_contrast=True):
        self.denoise = denoise
        self.normalize_contrast = normalize_contrast

    def load_image(self, image_path):
        image_path = Path(image_path)
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if gray is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        return gray

    def process(self, image_path):
        gray = self.load_image(image_path)
        processed = gray.copy()

        if self.denoise:
            processed = cv2.GaussianBlur(processed, (3, 3), 0)

        if self.normalize_contrast:
            processed = cv2.equalizeHist(processed)

        return gray, processed

    def save_image(self, image, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)

