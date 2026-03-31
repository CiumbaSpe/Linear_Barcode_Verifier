# Linear Barcode Verifier

This project is a small prototype for localizing and verifying linear barcodes from grayscale images.

The pipeline is split into a few simple steps:

- `Preproc.py`: image loading and optional preprocessing
- `Localizer.py`: finds the barcode region and estimates its geometry
- `Verifier.py`: rectifies the barcode ROI and computes verification metrics from multiple scan profiles
- `main.py`: runs either a single test image or the full dataset and saves the results

The current implementation focuses on two metrics:

- `contrast`
- `modulation`

Results from the full dataset are written to `results.csv`.

## Run

For a single image, set `IMAGE_NAME` in [settings.py] and keep `single_test()` enabled in [main.py]
For the whole dataset, comment `single_test()` and uncomment `full_test()` in [main.py]
