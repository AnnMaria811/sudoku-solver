from dataclasses import dataclass


@dataclass
class PipelineConfig:
    warp_size: int = 900                      # size of the square output grid (pixels)
    gaussian_kernel_size: int = 7             # blur kernel size for noise removal
    adaptive_block_size: int = 41             # neighbourhood size for adaptive threshold
    adaptive_c: int = 7                       # constant subtracted in adaptive threshold
    morphological_kernel_size: int = 3        # kernel for morphological close operation
    contour_epsilon_ratio: float = 0.02       # how aggressively to simplify contours
    min_board_area_ratio: float = 0.1         # grid must cover at least 10% of the image
    cell_margin_ratio: float = 0.14           # margin to strip from each cell before OCR
    blank_filled_ratio_threshold: float = 0.012  # cells below this fill ratio are empty
    min_component_area_ratio: float = 0.015   # ignore tiny specks inside cells
    digit_canvas_size: int = 32               # size of normalised digit tiles
    template_font_scale: float = 1.2
    template_thickness: int = 2
    ocr_min_score: float = 0.45              # minimum Dice score to accept a digit match
    ocr_min_accept_score: float = 0.50       # lower bound for secondary candidates
    ocr_score_margin: float = 0.08           # allow candidates within this margin of the best score
