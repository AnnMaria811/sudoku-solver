import cv2
import numpy as np


def generate_digit_templates(size):
    templates = {}
    canvas_size = size * 2

    for digit in range(1, 10):
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        text = str(digit)
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 5)
        x = (canvas_size - text_size[0]) // 2
        y = (canvas_size + text_size[1]) // 2
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 5, lineType=cv2.LINE_AA)
        _, canvas = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY)
        templates[digit] = centre_on_canvas(canvas, size)

    return templates


def load_templates(templates_dir, size):
    if templates_dir is not None and templates_dir.exists():
        templates = {}
        for digit in range(1, 10):
            for ext in ("png", "jpg", "jpeg"):
                path = templates_dir / f"{digit}.{ext}"
                if not path.exists():
                    continue
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                templates[digit] = centre_on_canvas(binary, size)
                break
        if templates:
            return templates

    return generate_digit_templates(size)


def centre_on_canvas(image, canvas_size):
    if image.size == 0 or np.count_nonzero(image) == 0:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    rows, cols = np.where(image > 0)
    digit = image[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]

    scale = (canvas_size - 8) / max(digit.shape)
    new_w = max(1, int(round(digit.shape[1] * scale)))
    new_h = max(1, int(round(digit.shape[0] * scale)))
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    y_start = (canvas_size - new_h) // 2
    x_start = (canvas_size - new_w) // 2
    canvas[y_start: y_start + new_h, x_start: x_start + new_w] = digit_resized

    return (canvas > 0).astype(np.uint8)


def is_empty_cell(cell, threshold=0.02):
    if cell.size == 0:
        return True
    return np.count_nonzero(cell) / cell.size <= threshold


def dice_score(template, cell):
    a = (template > 0).astype(np.uint8)
    b = (cell > 0).astype(np.uint8)
    overlap = int(np.count_nonzero(a & b))
    total = int(np.count_nonzero(a)) + int(np.count_nonzero(b))
    if total == 0:
        return 0.0
    return (2.0 * overlap) / total


def match_digit(cell, templates, min_score=0.45):
    if is_empty_cell(cell):
        return 0

    best_digit = 0
    best_score = 0.0

    for digit, template in templates.items():
        if template.shape != cell.shape:
            resized = cv2.resize(cell.astype(np.uint8), (template.shape[1], template.shape[0]), interpolation=cv2.INTER_NEAREST)
            cell_bin = (resized > 0).astype(np.uint8)
        else:
            cell_bin = (cell > 0).astype(np.uint8)

        score = dice_score(template, cell_bin)
        if score > best_score:
            best_score = score
            best_digit = digit

    if best_score < min_score:
        return 0

    return best_digit
