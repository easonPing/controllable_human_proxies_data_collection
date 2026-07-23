"""Launch the single-window human+AI Moving Out data collector."""

import argparse
import os
import sys


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from moving_out.collection.app import run_collection_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect one Moving Out human+AI session."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="saved_datasets",
        help="Directory for versioned session JSON files.",
    )
    parser.add_argument(
        "--ai-seed",
        type=int,
        default=None,
        help="Optional seed for reproducible AI-model actions.",
    )
    parser.add_argument(
        "--ai-timeout",
        type=float,
        default=2.0,
        help=(
            "Maximum in-flight AI inference time before the session is "
            "aborted (default: 2). Environment steps continue while an AI "
            "response is pending; models may enforce a higher minimum."
        ),
    )
    parser.add_argument(
        "--input",
        choices=["joystick", "keyboard"],
        default="joystick",
        help="Human input device (default: joystick).",
    )
    args = parser.parse_args()
    if args.ai_timeout <= 0:
        parser.error("--ai-timeout must be positive")

    # Environment frames are RGB and are handed directly to Pygame. Do not
    # apply the BGR conversion that OpenCV display paths require.
    run_collection_app(
        output_dir=args.output,
        ai_seed=args.ai_seed,
        ai_timeout=args.ai_timeout,
        input_mode=args.input,
    )


if __name__ == "__main__":
    main()
