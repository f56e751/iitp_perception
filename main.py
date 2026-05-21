#!/usr/bin/env python3
"""Single entry point for the IITP perception module.

Usage:
    python main.py <command> [options]

Commands:
    live      RealSense live capture + detection + MJPEG stream (results_local/)
    capture   RealSense capture only -- save frames, no detection
    batch     Detection on a folder of images (<dir>/images/)
    eval      Per-class score evaluation on saved images (scores.csv)
    group     Cluster scores.csv detections into per-object CSVs
    track     Cluster detections into motion tracks
    analyze   Compare track predictions against true labels
    correct   Apply manual label corrections to grouped objects

`python main.py <command> -h` shows options for that command.
Camera commands (live, capture) require the iitp_local image; the rest run
in the base grounded_sam image.
"""
import sys


def _live():
    from capture_and_detect import main
    return main()


def _capture():
    from perception_eval.capture_only import main
    return main()


def _batch():
    from iitp_object_detector import main
    return main()


def _eval():
    from perception_eval.eval_detector import main
    return main()


def _group():
    from perception_eval.group_by_object import main
    return main()


def _track():
    from perception_eval.group_by_track import main
    return main()


def _analyze():
    from perception_eval.analyze_tracks import main
    return main()


def _correct():
    from perception_eval.apply_corrections import main
    return main()


COMMANDS = {
    "live": _live,
    "capture": _capture,
    "batch": _batch,
    "eval": _eval,
    "group": _group,
    "track": _track,
    "analyze": _analyze,
    "correct": _correct,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0 if len(sys.argv) >= 2 else 1

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"unknown command: {cmd!r}\n", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        return 2

    # Hand the remaining args to the subcommand's own argparse.
    sys.argv = [f"{sys.argv[0]} {cmd}", *sys.argv[2:]]
    rc = COMMANDS[cmd]()
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
