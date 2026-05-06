from dataclasses import dataclass


@dataclass
class PipelineConfig:
    gaussian_kernel_size: int = 7
    adaptive_block_size: int = 41
    adaptive_c: int = 7
    morphological_kernel_size: int = 3
    contour_epsilon_ratio: float = 0.02
    min_board_area_ratio: float = 0.1
    warp_size: int = 900
    cell_margin_ratio: float = 0.14
    blank_filled_ratio_threshold: float = 0.012
    min_component_area_ratio: float = 0.015
    digit_canvas_size: int = 32
    ocr_min_score: float = 0.45
