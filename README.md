# Sudoku Solver

Sudoku Solver — Project Report
RCSS 5243 — Image Analysis and Computer Vision
Milestone 1 Submission · Ann Maria ID 900193992

1.  Introduction
The goal of this project is to build a program that can look at a photo of a Sudoku puzzle and solve it automatically. That means finding the grid in the image, reading the digits that are already filled in, and computing the missing ones. It sounded straightforward at first, but there ended up being quite a few steps involved, and some parts were a lot harder than expected, especially getting reliable digit recognition without any machine learning.
This report covers Milestone 1, which focuses on the image processing half of the pipeline: preprocessing the raw photo, isolating the outer frame of the grid, detecting its four corners, and warping the image into a clean top-down square. The OCR and solving stages are also included since the full pipeline was run on all test cases, but those will be the main focus of Milestone 2.

3.  Problem Definition
A typical photo of a Sudoku puzzle is noisy, unevenly lit, and taken from an angle. Before anything useful can be extracted, the image has to be cleaned up and geometrically corrected. The specific challenges are:
•	Lighting non-uniformity — shadows, glare, and uneven exposure can make parts of the grid appear much darker or lighter than others.
•	Perspective distortion — photos taken from any angle other than directly above will cause the grid to appear as a trapezoid rather than a square.
•	Noise — camera sensor noise and compression artefacts show up as random pixel variations that can interfere with edge detection and thresholding.
•	Digit variability — printed and handwritten digits vary significantly in stroke width, font, and size depending on the puzzle source.
The pipeline needs to handle all of these before any digit recognition can take place. The project specification also explicitly requires the OCR to be done with template matching rather than machine learning, which is an interesting constraint, it forces a more hands-on understanding of what makes recognition hard in the first place.

4.  Procedure
The whole system is wired together in a single pipeline (pipeline.py) that runs through five stages in order. Each stage saves output images to a folder, which makes it easy to visually inspect what happened at every step.

3.1  Image Preprocessing
The raw BGR image is first converted to grayscale. After that, Contrast Limited Adaptive Histogram Equalisation (CLAHE) is applied with a clip limit of 2.0 over an 8×8 tile grid. Standard global histogram equalisation can over-amplify noise in flat regions, but CLAHE applies equalisation locally within tiles and clips the contrast to a set limit, which keeps the bright and dark regions of the image balanced without making the noise worse.
The normalised image is then blurred with a 7×7 Gaussian kernel to suppress high-frequency noise before binarisation. Binarisation is done with an adaptive Gaussian threshold (block size 41, constant C = 7) in inverse polarity, so that dark lines on a light background become white foreground pixels. A 3×3 morphological closing operation is applied afterwards to bridge any small gaps that thresholding may have introduced in the grid lines.
3.2  Outer Frame Isolation
The binarised image is passed to cv2.findContours with RETR_EXTERNAL to get only the outermost contours. Contours below 10% of the total image area are discarded as noise. For each remaining contour, the Ramer–Douglas–Peucker algorithm approximates a polygon with ε = 2% of the contour perimeter. If this gives exactly four vertices, those are used directly as the grid corners; otherwise the minimum bounding rectangle is taken as a fallback.
To score each candidate, a penalty is applied based on how far the aspect ratio deviates from 1:1 — a real Sudoku grid should be roughly square. The contour with the highest penalised area score is selected as the board boundary.

3.3  Corner Identification
Once the best quadrilateral is found, its four corner points need to be ordered consistently for the perspective transform to work correctly. The ordering is done using coordinate sums and differences: the top-left corner has the smallest sum of (x + y), the bottom-right has the largest, the top-right has the smallest difference (x − y), and the bottom-left has the largest. Each detected corner is drawn on an overlay image as a red circle with an index label.

3.4  Perspective Warp (Grid Straightening)
With the four ordered corners known, cv2.getPerspectiveTransform maps them to a fixed 900×900 pixel square canvas. The image is then warped with cv2.warpPerspective, producing a top-down, undistorted view of just the puzzle grid. The inverse transformation matrix is also stored so that solved digits can be projected back onto the original image coordinate space if needed later.

3.5  OCR — Cell Extraction and Digit Matching
The warped 900×900 grid is divided into 81 equal cells (100×100 px each). A margin is cropped from each cell to exclude grid lines, and the largest connected component in the remaining region is extracted and centred on a 32×32 canvas. This centred binary patch is then compared against nine synthetic digit templates (generated with OpenCV's built-in Hershey Simplex font) using the Dice similarity coefficient:
Dice(A, B)  =  2 × |A ∩ B|  /  (|A| + |B|)
If the top score is below 0.45 the cell is treated as blank. Otherwise, a short list of close candidates is assembled and passed to a backtracking step (resolve_candidate_board) that checks each candidate against Sudoku row, column, and box constraints to find the most globally consistent assignment before passing anything to the solver.

3.6  Solver
The solver uses a standard recursive backtracking algorithm. It scans for the first empty cell, tries each digit 1–9, checks whether the placement is valid in the current row, column, and 3×3 box, and recurses. If it reaches a dead end it backtracks. The board is validated before the solver starts — if the OCR-produced givens already contain a duplicate, the pipeline raises an error instead of running the solver on a broken board.

 
Figure 1 — Four key stages: (a) original input, (b) CLAHE-normalised, (c) detected corners, (d) perspective-warped output.
 
Figure 2 — Grid isolation: binary mask (left) and contour overlay (right) for test case 01.
4.  Experimental Results
The full pipeline was run on all 16 provided test cases. The table below shows the result at each stage. For Milestone 1, the relevant columns are whether preprocessing and the warp completed successfully — OCR and solving results are shown for completeness but will be the focus of Milestone 2.
Test
Case	M1 Stages
(preprocess → warp)	Full Pipeline
(incl. OCR & solve)	Mean OCR
Confidence
01	Pass	Solved	0.243
02	Pass	Solved	0.266
03	Pass	OCR error	—
04	Pass	Solved	0.290
05	Pass	OCR error	—
06	Pass	OCR error	—
07	Pass	Solved	0.316
08	Detection failed	Detection failed	—
09	Pass	Solved	0.256
10	Pass	OCR error	—
11	Pass	Solved	0.278
12	Pass	OCR error	—
13	Pass	OCR error	—
14	Pass	Solved	0.286
15	Pass	OCR error	—
16	Pass	OCR error	—

Milestone 1 stages passed for 15 out of 16 test cases. Test case 08 failed at contour detection — the image appears to have very heavy glare, preventing a clean quadrilateral from being found in the binary image. Of the 15 cases that reached the full pipeline, 7 were solved end-to-end (44%). The other 8 produced OCR errors due to duplicate digit assignments, which is a known issue described in the next section.

 
Figure 3 — Test case 01: raw input (left) and perspective-corrected warped output (right).
 
Figure 4 — The 7 fully-solved test cases showing input and solution overlay side by side.

5.  Known Issues and Fixes Applied
When running on the full test set, the solved output sometimes showed a digit appearing more than once in the same row or column — which is obviously wrong for a Sudoku. This is not a solver bug; it comes from errors earlier in the pipeline being silently passed forward. After looking through the code, two root causes were identified:
Root Cause 1 — OCR Misreads
The template matching uses synthetic digit templates generated with OpenCV's Hershey Simplex font. Real puzzle digits especially printed ones with serifs, or slightly skewed handwritten ones — look quite different, so the Dice scores tend to be low and the wrong digit can get selected. This is the core limitation of template matching without ML and is expected at this stage.
Root Cause 2 — Silent Fallback to Unvalidated Board
When resolve_candidate_board could not find a constraint-consistent assignment, the original code fell back silently to the raw OCR guesses. Those guesses could already contain duplicate digits in the same row or column, so the solver would produce a valid completion of the wrong board — matching its own (wrong) givens perfectly but not the real printed digits.

Fixes Applied
•	Removed the silent fallback. The pipeline now raises a PipelineError with a clear message if resolve_candidate_board fails, instead of passing invalid data forward.
•	OCR thresholds moved to config. The three match thresholds (minimum score, acceptance floor, score margin) are now PipelineConfig fields so they can be tuned without hunting through the code. The minimum score was also raised from 0.45 to 0.55.
•	Added a givens-preservation check. After solving, the code verifies that every OCR-flagged given still holds the same digit in the solved board catching any future regression immediately.
•	Overlay now colour-codes OCR givens vs. solver-filled cells (blue vs. green), making it easy to spot at a glance which digits were read from the image and which were computed.
What still needs work is the root cause: the mismatch between synthetic font templates and real puzzle digit appearances. A practical fix for Milestone 2 would be to supply hand-cropped templates from real puzzle images in an assets/ folder, or to collect multiple samples per digit and take the best match across all of them.

6.  Conclusion
The Milestone 1 requirements are fully met. The preprocessing pipeline reliably normalises contrast and produces clean binary images across varied lighting. Contour-based isolation finds the grid boundary in 15 of 16 test images, and the perspective warp produces a properly rectified 900×900 output in every detected case.
Working on this made it very clear why each step matters. Skipping CLAHE and going straight to a global threshold caused obvious failures on images with uneven lighting. Getting the corner ordering right also took more debugging than expected the perspective warp silently produces garbage if the four points are fed in the wrong order.
For Milestone 2, the OCR stage is the main thing to fix. The Dice-score template matching works in easy cases but falls apart with real puzzle fonts. Providing better templates and refining the candidate resolution step should push the end-to-end success rate significantly higher. The modular structure of the codebase (each stage is a separate file with typed inputs and outputs) makes it straightforward to swap in an improved matcher without touching the rest of the pipeline.

References
Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.
Gonzalez, R. C. & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson.
Ramer, U. (1972). An iterative procedure for the polygonal approximation of plane curves. Computer Graphics and Image Processing, 1(3), 244–256.

