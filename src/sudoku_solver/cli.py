import argparse
import sys
from pathlib import Path

from .pipeline import run_pipeline


STAGES = ("preprocess", "detect", "warp", "ocr", "solve")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sudoku-solver")
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve = subparsers.add_parser("solve", help="process a Sudoku image and solve it")
    solve.add_argument("image_path", type=Path, help="path to the input image")
    solve.add_argument("--out", dest="output_dir", type=Path, required=True, help="folder to save output files")
    solve.add_argument("--stage", dest="stop_after_stage", choices=STAGES, help="stop after this pipeline stage")
    solve.add_argument("--templates", dest="templates_dir", type=Path, help="folder containing digit template images")
    solve.add_argument("--no-save-stages", dest="save_stages", action="store_false", default=True)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "solve":
        try:
            run_pipeline(
                image_path=args.image_path,
                output_dir=args.output_dir,
                stop_after_stage=args.stop_after_stage,
                save_stages=args.save_stages,
                templates_dir=args.templates_dir,
            )
            return 0
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    parser.error(f"unknown command: {args.command}")
    return 2
