# Sudoku Solver — RCSS 5243 Project

This project takes a photo of a sudoku puzzle, reads the digits using pattern matching, and solves the puzzle. It was built as part of the RCSS 5243 Image Analysis and Computer Vision course.

## Demo

> [Watch the demo video](https://youtu.be/PLACEHOLDER)

---

## How it works

The program processes the image through five stages in order. Each stage builds on the previous one.

### Stage 1 – Preprocessing

Before we can detect anything, the image needs to be cleaned up. Photos taken with a phone can have uneven lighting, shadows, noise, etc. We handle this in a few steps:

1. **Convert to grayscale** — we only need brightness values, not colour.
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalisation) — this fixes uneven lighting. Without it, dark corners or bright reflections make the threshold step fail.
3. **Gaussian blur** — smooths out small specks of noise before thresholding.
4. **Adaptive threshold** — converts the image to black and white. We use adaptive (not global) thresholding because different parts of the image can have different lighting. A single global cutoff doesn't work well here.
5. **Morphological close** — fills tiny gaps in the grid lines so they look solid and connected. This is important for the contour detection step.

### Stage 2 – Finding the Grid

We use `cv2.findContours` to find all the outlines in the binary image, then filter them:

- The grid must cover at least 10% of the image (anything smaller is probably not the puzzle).
- We simplify each contour with `cv2.approxPolyDP` to get a polygon shape.
- We want a quadrilateral (4 corners). If we don't get 4 corners, we fall back to the minimum bounding rectangle.
- We take the largest valid quadrilateral — that should be the outer border of the sudoku grid.

Once we have the 4 corners, we sort them into a consistent order (top-left, top-right, bottom-right, bottom-left) using a simple trick: `x + y` is smallest for the top-left corner and largest for the bottom-right. The difference `y - x` separates the other two.

### Stage 3 – Straightening the Grid

Even if the photo was taken at an angle, we can correct it using a perspective transform. We tell OpenCV to map the four detected corners to the four corners of a 900×900 pixel square. The result is a flat, top-down view of the grid.

After warping, we run preprocessing again on the warped image to get a clean binary version for OCR.

### Stage 4 – Reading the Digits (OCR)

This is the most interesting part. We don't use any machine learning here — just image template matching.

**Cell extraction:**
We divide the 900×900 grid into an 9×9 grid of 100×100 cells. We strip a 14-pixel margin from each edge to remove the grid lines. Inside each cell, we find the largest connected blob (using `cv2.connectedComponentsWithStats`) and throw away any tiny specks that are probably just noise. The remaining blob — if there is one — is the digit.

**Template generation:**
We draw each digit 1–9 using OpenCV's built-in `HERSHEY_SIMPLEX` font and use those as our reference templates. We also crop and centre each template on a fixed 32×32 canvas so the comparison is fair.

**Matching:**
Each cell image is also centred on a 32×32 canvas, then compared against all 9 templates using the **Dice similarity coefficient**:

```
Dice(A, B) = 2 × |A ∩ B| / (|A| + |B|)
```

This measures the overlap between two binary images as a score from 0 to 1. We use Dice instead of just counting matching pixels because it accounts for the sizes of both images — a narrow digit like "1" won't score badly just because it has fewer pixels than a wide digit like "8".

If the best score is above 0.45, we accept it as a match. If nothing clears that threshold, the cell is treated as empty (0).

### Stage 5 – Solving

We solve the puzzle using **backtracking**: find the first empty cell, try placing each digit 1–9, check if it's valid (no duplicates in the same row, column, or 3×3 box), and recurse. If we get stuck we undo the last move and try the next digit.

The solution is drawn back on the warped grid image — given digits in blue, solver-filled digits in green.

---

## Output files

After running, the output folder will contain:

| File | What it is |
|------|-----------|
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

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.14+, NumPy 2.4+, OpenCV 4.10+.

## Running it

```bash
# solve a puzzle image
sudoku-solver solve path/to/puzzle.jpg --out output/

# stop at a specific stage (useful for debugging)
sudoku-solver solve puzzle.jpg --out output/ --stage warp

# use your own digit template images
sudoku-solver solve puzzle.jpg --out output/ --templates assets/digit_templates/
```

## Custom templates

If the built-in templates don't work well for your puzzle's font, you can provide your own. Place grayscale images named `1.png`, `2.png`, … `9.png` in `assets/digit_templates/`. Each image should show just the digit on a plain background. The pipeline will binarise them automatically.

## Running the tests

```bash
pytest tests/ -v
```

---

## References

1. OpenCV — Adaptive Thresholding tutorial: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
2. OpenCV — Morphological Operations: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
3. OpenCV — Contours tutorial: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
4. OpenCV — Perspective Transform (`getPerspectiveTransform`, `warpPerspective`): https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
5. Zuiderveld, K. (1994). *Contrast Limited Adaptive Histogram Equalization*. Graphics Gems IV. — this is what CLAHE is based on.
6. Dice, L. R. (1945). *Measures of the Amount of Ecologic Association Between Species*. Ecology, 26(3), 297–302. — the formula we use for template matching.
7. Norvig, P. (2011). *Solving Every Sudoku Puzzle*: https://norvig.com/sudoku.html — helpful reference for the backtracking solver.
8. "Sudoku Grab" blog post (the real-world inspiration for this project): http://blog.francoisjacquet.com/2011/09/14/solving-a-sudoku-puzzle-from-a-photograph/
