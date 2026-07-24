# DecodeVision

DecodeVision is an inspectable optical character recognition workspace built
for DecodeLabs Artificial Intelligence Project 4. It turns a document image
into validated text through a visible sequence of OpenCV transformations and a
native Tesseract OCR adapter.

The project chooses the **OCR execution path** from the supplied blueprint and
implements its complete milestone:

- grayscale conversion and Gaussian noise reduction;
- adaptive Gaussian or Otsu thresholding;
- safe foreground-based deskewing;
- Tesseract page segmentation modes 3, 6, 7 and 11;
- word-level text, confidence and bounding-box coordinates;
- a configurable confidence gate that defaults to the required 80%;
- visual evidence for accepted and rejected words;
- TXT and JSON exports;
- an offline sample document and a CLI workflow.

## Architecture

```text
PNG/JPG bytes
     │
     ▼
decode + image validation
     │
     ▼
grayscale → Gaussian blur → binary threshold → deskew
     │
     ▼
Tesseract image_to_data
     │
     ▼
words + coordinates + confidence
     │
     ▼
80% gate → validated text + bounding-box evidence
```

The interface never calls Tesseract directly. `app.py` delegates to the
preprocessing pipeline and the native adapter, then renders domain results.
This keeps the recognition boundary testable and prevents UI logic from
fabricating model output.

See [Project 4 requirement traceability](docs/project-4-requirements.md) for the
complete brief-to-code mapping.

## Install

### 1. Python dependencies

From `04-decodevision-ocr`:

```powershell
python -m pip install -r requirements.txt
```

### 2. Native Tesseract engine

On Windows:

```powershell
winget install -e --id UB-Mannheim.TesseractOCR
```

Restart PowerShell after installation. DecodeVision searches `PATH` and common
Windows installation folders automatically. For a custom location:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Tesseract is intentionally not bundled in Git because it is a native
operating-system dependency.

## Run

```powershell
python scripts/generate_sample.py
python -m streamlit run app.py
```

The OpenCV stages remain available even when Tesseract is missing. Recognition
is enabled only when the real native engine is detected.

## CLI

```powershell
python scripts/run_ocr.py data/samples/decodevision_invoice.png `
  --threshold 80 `
  --psm 6 `
  --threshold-mode adaptive
```

Exit code `0` means the validated OCR output passed the active confidence gate;
`1` means recognition completed below it; `2` means setup or input failed.

## Test

The automated suite does not require the native Tesseract executable:

```powershell
python -m unittest discover -s tests -v
```

It validates image decoding, preprocessing stages, threshold modes, confidence
parsing and filtering, orchestration, bounding boxes and the four Project 4
milestone checks.

## Project structure

```text
04-decodevision-ocr/
├── app.py
├── data/samples/decodevision_invoice.png
├── scripts/
│   ├── generate_sample.py
│   └── run_ocr.py
├── src/vision/
│   ├── models.py
│   ├── ocr.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   └── validation.py
├── tests/test_decodevision.py
└── docs/project-4-requirements.md
```

## Scope

DecodeVision is an educational OCR pipeline, not a document-authentication,
financial or identity-verification system. Confidence is the OCR engine's
self-assessment, not a guarantee of correctness.

## Author

Jean Franck Loa Rojas
