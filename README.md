# Sudoku Solver — Ann Maria Project


This project takes a photo of a Sudoku puzzle, reads the digits using template-based pattern matching, and solves the puzzle. 

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Definition](#2-problem-definition)
3. [How It Works](#3-how-it-works)
4. [Experimental Results](#4-experimental-results)
5. [Analysis](#5-analysis)
6. [Known Limitations](#6-known-limitations)
7. [Conclusion](#7-conclusion)
8. [Setup & Usage](#8-setup--usage)
9. [Output Files](#9-output-files)
10. [References](#10-references)

---

## 1. Introduction

The goal of this project is to build a program that can look at a photo of a Sudoku puzzle and solve it automatically — finding the grid, reading the digits that are already filled in, and computing the missing ones. It sounds straightforward at first, but there are quite a few steps involved, and some parts are harder than expected, especially getting reliable digit recognition without any machine learning.

The pipeline covers five stages: preprocessing the raw photo, isolating the outer frame of the grid, detecting its four corners, warping the image into a clean top-down square, reading the digits via template matching, and solving the puzzle with a backtracking algorithm. No machine learning is used at any point — digit recognition is done entirely through classical computer vision and pixel-level template comparison, as required by the project specification.

---

## 2. Problem Definition

A typical photo of a Sudoku puzzle is noisy, unevenly lit, and taken at an angle. The system has to handle all of this before any digit recognition can happen. The specific challenges are:

- **Lighting non-uniformity** — shadows and glare cause different regions of the grid to appear at very different brightness levels.
- **Perspective distortion** — photos taken at an angle make the grid appear as a trapezoid rather than a square, which needs to be corrected before cells can be extracted accurately.
- **Grid line interference** — the lines of the Sudoku grid overlap with the digit pixels inside each cell, so they need to be excluded during OCR.
- **Digit variability** — the same digit looks quite different across puzzle sources: different fonts, stroke widths, sizes, and print quality all affect matching.
- **Template mismatch** — since templates are generated synthetically using OpenCV's built-in font, there is an inevitable gap between the template shapes and real printed digits, which limits recognition accuracy.

The project specification also specifically requires OCR to be implemented without machine learning — only pattern matching. This constraint is useful: building a basic matcher makes it very clear why ML-based approaches outperform template matching in practice.

---

## 3. How It Works

The program processes the image through five stages in order. Each stage saves output images to a folder, making it easy to inspect what happened at every step.

### Stage 1 — Preprocessing (`preprocess.py`)

Before anything else, the image needs to be cleaned up. Photos taken with a phone can have uneven lighting, shadows, and noise. The preprocessing steps are:

1. **Convert to grayscale** — only brightness values are needed, not colour.
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalisation, clip limit 2.0, 8×8 tile grid) — fixes uneven lighting. Unlike global histogram equalisation, CLAHE works on small tiles independently and clips the amplification at a fixed limit, so it does not over-brighten already-light areas or introduce heavy noise. Without it, dark corners or bright reflections make the threshold step fail.
3. **Gaussian blur** (7×7 kernel) — smooths out small specks of noise before thresholding.
4. **Adaptive threshold** (block size 41, C = 7, inverse polarity) — converts the image to black and white. Adaptive thresholding is used instead of a global cutoff because different parts of the image can have very different lighting levels.
5. **Morphological close** (3×3 kernel) — fills tiny gaps in the grid lines so they look solid and connected. This matters because broken lines confuse the contour detector in the next stage.

### Stage 2 — Grid Detection (`grid_detection.py`)

The binarised image is searched for the Sudoku grid boundary:

- `cv2.findContours` with `RETR_EXTERNAL` retrieves only the outermost contours.
- Any contour covering less than 10% of the image area is discarded as noise.
- Each qualifying contour is approximated with `cv2.approxPolyDP` (ε = 2% of the perimeter) to get a polygon shape.
- A quadrilateral (4 corners) is preferred. If the approximation does not produce exactly 4 vertices, the minimum bounding rectangle is used as a fallback.
- The largest valid quadrilateral by area is taken as the outer border of the puzzle.

Once the 4 corners are found, they are sorted into a consistent order — top-left, top-right, bottom-right, bottom-left — using coordinate arithmetic: `x + y` is smallest at the top-left and largest at the bottom-right; `x − y` separates the other two.

### Stage 3 — Perspective Warp (`warp.py`)

`cv2.getPerspectiveTransform` maps the four detected corners to the four corners of a fixed 900×900 pixel square canvas. The image is then warped with `cv2.warpPerspective`, giving a flat, top-down view of just the puzzle grid regardless of the original camera angle. After warping, preprocessing is run again on the warped image to produce a clean binary version for cell extraction and OCR.

### Stage 4 — OCR: Cell Extraction and Digit Matching (`cell_extraction.py`, `ocr.py`)

This is the most technically involved part. No machine learning is used — only image template matching.

**Cell extraction:**  
The warped 900×900 binary image is divided into an 9×9 grid of 100×100 cells. A margin of approximately 14 px is stripped from each edge to remove the grid lines. Inside the trimmed region, `cv2.connectedComponentsWithStats` finds all connected blobs. The largest blob above a minimum area threshold is kept as the digit candidate; anything smaller is treated as noise and discarded. The retained blob is centred on a fresh 32×32 canvas using `centre_on_canvas()`, which scales the glyph to fill the canvas (minus small padding) and places it at the centre. This normalisation is what makes fair comparison against templates possible.

**Template generation:**  
Templates for digits 1–9 are generated synthetically using OpenCV's `FONT_HERSHEY_SIMPLEX` (scale 2.2, thickness 5), then binarised and centred on the same 32×32 canvas. Custom templates can also be supplied — see [Custom Templates](#custom-templates) below.

**Matching:**  
Each 32×32 cell image is compared against all nine templates using the **Sørensen–Dice coefficient**:

```
Dice(A, B) = 2 × |A ∩ B| / (|A| + |B|)
```

This measures the overlap between two binary images as a score from 0 to 1. Dice is used instead of raw pixel counting because it normalises by the size of both images — a narrow digit like "1" has far fewer foreground pixels than "8", so raw overlap would systematically favour wider templates. Dice removes that bias.

If the best score is above 0.45, that digit is accepted. If nothing clears that threshold, the cell is treated as empty (0).

### Stage 5 — Solver (`solver.py`)

The board is validated before solving. `validate_board` checks that the grid is 9×9, every value is in 0–9, and no digit appears twice in any row, column, or 3×3 box. If the OCR output already violates any constraint, an `InvalidBoardError` is raised immediately rather than running the solver on a broken board.

Solving uses standard recursive backtracking: find the first empty cell, try each digit 1–9, check validity, recurse. If no digit works, backtrack to the previous cell and try the next candidate. The algorithm is correct and complete — it will always find a solution if one exists.

The solution is drawn on the warped grid image: OCR-read givens in blue, solver-filled digits in green.

---

## 4. Experimental Results

The full pipeline was run on all 16 provided test cases. A case is marked **Solved** only if the OCR output contained no duplicate digits and the solver found a valid completion. Cases marked **OCR error** reached the recognition stage but produced a board with at least one duplicate digit, which fails validation and prevents solving.

| Test Case | Preprocess & Warp | Digits Recognised | Full Pipeline | Notes |
|:---------:|:-----------------:|:-----------------:|:-------------:|-------|
| 01 | ✅ Pass | 26 / 81 | ❌ OCR error | Duplicate in row 6, col 3, col 9 |
| 02 | ✅ Pass | 29 / 81 | ❌ OCR error | Duplicates in 3 rows, 3 cols |
| 03 | ✅ Pass | 35 / 81 | ❌ OCR error | Duplicates in 4 rows, 3 cols |
| 04 | ✅ Pass | 30 / 81 | ❌ OCR error | Duplicate in rows 6, 8, 9 |
| 05 | ✅ Pass | 27 / 81 | ❌ OCR error | Duplicates in 4 rows, 4 cols |
| 06 | ✅ Pass | 30 / 81 | ❌ OCR error | Duplicate in row 7, 3 cols |
| 07 | ✅ Pass | 31 / 81 | ❌ OCR error | Duplicate in row 1, col 6, col 7 |
| 08 | ❌ Detect fail | — | ❌ Detect fail | Heavy glare — grid contour not found |
| 09 | ✅ Pass | 28 / 81 | ❌ OCR error | Duplicates in 4 rows, 5 cols |
| 10 | ✅ Pass | 37 / 81 | ❌ OCR error | Heavy over-recognition, many duplicates |
| 11 | ✅ Pass | 32 / 81 | ❌ OCR error | Duplicate in row 2, col 7, col 9 |
| 12 | ✅ Pass | 37 / 81 | ❌ OCR error | Widespread duplicates — likely warp misalignment |
| 13 | ✅ Pass | 26 / 81 | ❌ OCR error | Duplicates in 4 rows, 5 cols |
| 14 | ✅ Pass | 32 / 81 | ❌ OCR error | Duplicates in 4 rows, 4 cols |
| 15 | ✅ Pass | 19 / 81 | ❌ OCR error | Low recognition rate |
| 16 | ✅ Pass | 20 / 81 | ❌ OCR error | Low recognition rate |

**Summary:** 15/16 test cases passed the preprocessing and warping stages. Test case 08 failed at contour detection due to strong glare preventing a clean quadrilateral from being found. For the OCR and solving stages, every recognised board contained at least one duplicate digit from OCR misreads, which fails validation and prevents solving. The pipeline correctly identifies this and raises a clear error rather than producing a silently wrong answer.

---

## 5. Analysis

### What works well

- **Preprocessing is robust.** CLAHE normalisation handles moderate uneven lighting well across the full test set. No test case failed due to a preprocessing issue.
- **Grid detection and warping work on 15/16 images.** The contour-based approach reliably finds the outer boundary and the perspective warp produces clean square grids.
- **Cell extraction is clean.** The connected-component approach successfully isolates digit blobs and excludes grid lines through margin cropping.
- **The solver is correct.** `validate_board` correctly rejects any board with duplicates before the solver runs, so no invalid puzzle was ever attempted. The backtracking algorithm is complete — it would solve any valid board given correct OCR input.
- **Failures are visible, not silent.** When OCR produces an invalid board, the pipeline raises a descriptive error. The `08_digit_crops/` folder alongside `09_recognized_grid.txt` makes it easy to trace exactly which cells were misread.

### Root cause of OCR failures

Every failure comes from the same source: the Dice-score template matcher misidentifies one or more digits, producing a board with duplicate digits that fails validation. There are two specific causes:

**Font mismatch.** The templates are rendered with OpenCV's Hershey Simplex font. Real puzzle digits — especially from printed newspapers or book puzzles — use serif or condensed fonts that look quite different at the binary level. The Dice score between a real digit and its Hershey equivalent can be low enough that the matcher picks the wrong digit. Pairs like 1 vs 7, 3 vs 8, and 6 vs 9 are particularly prone to confusion.

**Cell alignment noise.** Even after the perspective warp, individual cell boundaries are not perfectly aligned with the grid lines. If the warp is slightly off, the margin-cropping step may either include part of a grid line (adding noise to the cell) or cut into the digit itself (removing pixels). Both distort the centred patch and lower the Dice score against the correct template.

### What this shows about template matching vs. ML

The 0% end-to-end success rate is a clear demonstration of why machine learning is used for OCR in practice. Template matching works when the input font closely matches the template font — it is essentially exact shape comparison. The moment the font differs, stroke weights vary, or digits are slightly rotated, accuracy collapses. A convolutional neural network learns features that are invariant to these variations through training on diverse examples. Building this basic matcher makes that contrast concrete in a way that just reading about it does not.

---

## 6. Known Limitations

**OCR accuracy — font mismatch.**  
The most impactful improvement would be replacing the synthetic Hershey templates with hand-cropped real templates taken from actual puzzle images. The pipeline already supports this: placing `1.png` … `9.png` in `assets/digit_templates/` causes `load_templates()` to use them instead of the generated ones. One clean crop per digit from a representative puzzle would likely push the success rate substantially higher.

**Grid detection for glare (test case 08).**  
Heavy glare washes out part of the grid boundary, leaving the contour detector unable to find a clean quadrilateral. A fallback using Otsu thresholding (which handles bimodal histograms well) or a Hough line transform to detect grid lines directly could recover these cases.

**OCR score threshold tuning.**  
The minimum Dice threshold (`ocr_min_score = 0.45`) is a single global value. The right value depends on the puzzle source. It is already in `PipelineConfig` for easy tuning — an extension would be to auto-calibrate it from the score distribution of the current image.

**Candidate resolution.**  
An earlier version included a `resolve_candidate_board` step that used Sudoku row/column/box constraints to choose between ambiguous OCR candidates (cells where multiple digits had similar scores). Reintroducing it would help cases where the top-scoring digit is wrong but a close second candidate is consistent with the rest of the board.

---

## 7. Conclusion

The pipeline implements all required components: preprocessing, grid isolation, corner detection, perspective warping, cell extraction, template-based OCR using the Dice similarity coefficient, and a validated backtracking solver. The code is modular — each stage is a separate file with clear inputs and outputs — and the pipeline saves intermediate images at every step, making failures easy to trace.

The preprocessing and geometry stages (Milestone 1) work correctly on 15 out of 16 test images. The OCR stage correctly classifies cells and measures match confidence using Dice, but the synthetic Hershey templates do not match real puzzle digit fonts closely enough to produce a duplicate-free board on any of the test cases. As a result, the solver is never reached — the board validator correctly rejects every recognised board before solving starts.

This outcome is informative rather than just a failure. It makes a concrete, measurable case for why machine-learning-based OCR replaced template matching: the template approach is brittle to font variation in a way that is difficult to engineer around without essentially re-implementing feature learning by hand. The pipeline is designed to support easy improvement — swapping in better templates or reintroducing candidate resolution are both one-file changes.

---

## 8. Setup & Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.14+, NumPy 2.4+, OpenCV 4.10+.

```bash
# Solve a puzzle image
sudoku-solver solve path/to/puzzle.jpg --out output/

# Stop at a specific stage (useful for debugging)
sudoku-solver solve puzzle.jpg --out output/ --stage warp

# Use your own digit template images
sudoku-solver solve puzzle.jpg --out output/ --templates assets/digit_templates/
```

### Custom templates

If the built-in templates don't work well for your puzzle's font, you can provide your own. Place grayscale images named `1.png`, `2.png`, … `9.png` in `assets/digit_templates/`. Each image should show just the digit on a plain background. The pipeline will binarise them automatically.

### Running the tests

```bash
pytest tests/ -v
```

---

## 9. Output Files

| File | What it is |
|------|------------|
| `01_original.png` | The input image |
| `02_preprocessed.png` | After CLAHE + blur + threshold |
| `03_grid_mask.png` | Filled mask of the detected grid |
| `04_lines_or_contour.png` | Detected grid outline on the original image |
| `05_corners_overlay.png` | The four corners marked with numbers |
| `06_warped_grid.png` | Perspective-corrected square grid |
| `07_cells_overlay.png` | Each cell coloured green (digit) or red (empty) |
| `08_digit_crops/cell_r_c.png` | Normalised 32×32 image for each cell |
| `09_recognized_grid.txt` | OCR result (0 = empty cell) |
| `10_solved_grid.txt` | The completed puzzle |
| `11_solution_overlay.png` | Solution drawn on the warped grid |
| `run_summary.json` | JSON summary of the run |

---

## 10. References

1. OpenCV — Adaptive Thresholding: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
2. OpenCV — Morphological Operations: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
3. OpenCV — Contours: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
4. OpenCV — Perspective Transform: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
5. Zuiderveld, K. (1994). *Contrast Limited Adaptive Histogram Equalization*. Graphics Gems IV, Academic Press.
6. Dice, L. R. (1945). *Measures of the Amount of Ecologic Association Between Species*. Ecology, 26(3), 297–302.
7. Ramer, U. (1972). *An iterative procedure for the polygonal approximation of plane curves*. Computer Graphics and Image Processing, 1(3), 244–256.
8. Gonzalez, R. C. & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
