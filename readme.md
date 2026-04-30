# Linear Barcode Verifier

This project reimplements part of the ISO/IEC 15416 verification workflow for linear barcodes.

## Pipeline 

The pipeline is split into a few steps:

- `Preproc.py`: image loading and optional preprocessing
- `Localizer.py`: finds the barcode region and estimates its geometry
- `Verifier.py`: rectifies the barcode ROI and computes verification metrics from multiple scan profiles
- `Grader.py`: assigns scanline and symbol grades from the computed metrics
- `main.py`: runs either a single test image or the full dataset and saves the results

## Metrics

The current implementation computes:

- `min_reflectance`: $R_{min}$
- `min_edge_contrast`: $EC_{min}$
- `contrast` =  $R_{max} - R_{min}$
- `modulation` = $100 \cdot \frac{EC_{min}}{\text{contrast}}$
- `defects` = $100 \cdot \frac{ERN_{max}}{\text{contrast}}$
- `numeric_grade`
- `grade`

where:

$$
ERN_i = R_{max,i} - R_{min,i}
$$

and $(ERN_{max})$ is the maximum ERN over all barcode elements and quiet zones.

Results from the full dataset are written to `results.csv`.

## Run

For a single image, set `IMAGE_NAME` in [`settings.py`](settings.py) and keep `single_test()` enabled in [`main.py`](main.py).

For the whole dataset, comment `single_test()` and uncomment `full_test()` in [`main.py`](main.py).