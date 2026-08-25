"""SAM3 TensorRT detection backend for the live streaming pipeline.

Returns exactly what ``iitp_object_detector.object_detector`` returns --
``(bounding_boxes, class_names, confidences)`` plus the pixel-space overlay
data main.py draws on the MJPEG frame -- so the :8080 contract (video,
/detections, /detections/stream, /latency) is unchanged and only the model
behind it differs.

The vendored runtime under ``SAM3-trt/`` imports tensorrt / pycuda at module
scope, so it is loaded lazily on first use: importing this module on a host
without the inference environment stays cheap, which keeps the unit tests
runnable there.
"""

import importlib.util
import os
from types import SimpleNamespace

import numpy as np

from bbox_projection import project_bounding_boxes

_ROOT = os.path.dirname(os.path.abspath(__file__))
SAM3_DIR = os.path.join(_ROOT, "SAM3-trt")
SERVE_MODULE_PATH = os.path.join(SAM3_DIR, "serve_trt_iitp_usls.py")

# The vendored runtime spells the cardboard category "Cardboard"; the stream
# and every offline tool in this repo use the lowercase form.
CATEGORY_ALIASES = {"Cardboard": "cardboard"}

# Fixed per-class ids so a class keeps its colour on the stream from frame to
# frame (the GroundingDINO engine numbered detections instead, so the colour of
# a given class changed whenever the detection count changed).
CLASS_IDS = {"transparent": 0, "metal": 1, "cardboard": 2}

_runtime = None


def load_runtime():
    """Import the vendored SAM3-trt server module (tensorrt/pycuda land here)."""
    global _runtime
    if _runtime is None:
        spec = importlib.util.spec_from_file_location(
            "sam3_serve_runtime", SERVE_MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _runtime = module
    return _runtime


def default_args(**overrides):
    """Runtime settings for the vendored ``RuntimeSegmenter``.

    Mirrors serve_trt_iitp_usls.py's argparse defaults for everything that does
    not depend on the runtime module itself.  The precision preset and the
    prompt specs are filled in by :class:`Sam3Detector` from the runtime, so
    this stays importable (and testable) without tensorrt.
    """
    weights = os.path.join(_ROOT, "weights")
    args = SimpleNamespace(
        vision_engine=os.path.join(
            weights, "SAM3-trt", "usls_engines_b2p2", "vision_b2_fp16.engine"
        ),
        text_engine=os.path.join(
            weights, "SAM3-trt", "usls_engines", "text_b6_fp16.engine"
        ),
        decoder_engine=os.path.join(
            weights, "SAM3-trt", "usls_engines", "decoder_b1_p32_fp16.engine"
        ),
        tokenizer=os.path.join(weights, "usls", "tokenizer.json"),
        image_size=1008,
        score_thr=0.4,
        mask_thr=0.0,
        process_area_thr=1000,
        process_prefer="smaller",
        merge_nms_iou=0.5,
        top_k=None,
        # Temporal voting assumes a known conveyor speed in px/frame; it stays
        # off until this installation's belt motion is measured.
        temporal_vote=False,
        conveyor_dx_px_per_frame=-1.0,
        conveyor_dy_px_per_frame=-68.0,
        temporal_iou_thr=0.20,
        temporal_vote_thr=0.50,
        temporal_current_weight=1.0,
        temporal_history_weight=0.65,
        temporal_decay=0.85,
        temporal_max_age=2,
        save_overlays=False,
        out_dir=os.path.join(_ROOT, "output", "usls_runtime_overlays"),
        prompt_specs=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def normalize_category(category):
    """Runtime category name -> the class name used across this repo."""
    return CATEGORY_ALIASES.get(category, category)


def instances_to_detections(instances, project, depth_np):
    """Adapt runtime instances into this repo's detection tuple + overlay data.

    ``instances`` is the list ``RuntimeSegmenter.infer`` returns: one dict per
    kept mask with ``category``, ``score`` and ``bbox_xyxy`` in the pixel coords
    of the image fed to the model (the centre crop).  They are re-ordered by
    score, highest first, so the stream labels the most confident object first.

    Returns ``(bounding_boxes, class_names, confidences, overlay)`` where
    ``overlay`` is ``(xyxy, class_ids, labels)`` in those same crop pixels.
    """
    ordered = sorted(instances, key=lambda inst: -float(inst["score"]))
    class_names = [normalize_category(inst["category"]) for inst in ordered]
    confidences = [float(inst["score"]) for inst in ordered]
    xyxy = np.asarray(
        [inst["bbox_xyxy"] for inst in ordered], dtype=np.float32
    ).reshape((-1, 4))
    class_ids = np.asarray(
        [CLASS_IDS.get(name, 0) for name in class_names], dtype=int
    )

    # Project the complete 2D box, corner order clockwise from top-left, the
    # same way the GroundingDINO engine does.
    bounding_boxes = project_bounding_boxes(xyxy, project, depth_np)

    labels = [
        f"{name} {score:.2f} "
        f"({sum(p[0] for p in box) / len(box):+.2f},"
        f"{sum(p[1] for p in box) / len(box):+.2f})m"
        for name, score, box in zip(class_names, confidences, bounding_boxes)
    ]
    return bounding_boxes, class_names, confidences, (xyxy, class_ids, labels)


EMPTY_OVERLAY = (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=int), [])


class Sam3Detector:
    """Per-frame SAM3 TensorRT detection with the engines loaded once.

    Pass ``segmenter`` to drive a stand-in in tests; otherwise the vendored
    ``RuntimeSegmenter`` is built from ``default_args(**overrides)``.
    """

    def __init__(self, segmenter=None, precision_preset=True, **overrides):
        self.args = default_args(**overrides)
        if segmenter is None:
            runtime = load_runtime()
            if self.args.prompt_specs is None:
                self.args.prompt_specs = runtime.DEFAULT_PROMPT_SPECS
            if precision_preset:
                runtime.apply_precision_preset(self.args)
            require_runtime_files(self.args)
            segmenter = runtime.RuntimeSegmenter(self.args)
        self.segmenter = segmenter
        # Latest frame's overlay data in centre-crop pixel coords:
        # (xyxy, class_ids, labels).  main.py shifts it onto the full frame.
        self.last_overlay = EMPTY_OVERLAY
        self.last_timing_ms = None

    def detect(self, color_np, depth_np, project, frame_delta=1.0):
        """Run one BGR frame through SAM3; same shape as object_detector()."""
        import cv2
        import PIL.Image

        image = PIL.Image.fromarray(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))
        result = self.segmenter.infer(image, frame_delta=frame_delta)
        self.last_timing_ms = result.get("timing_ms")
        bounding_boxes, class_names, confidences, overlay = instances_to_detections(
            result["instances"], project, depth_np
        )
        self.last_overlay = overlay
        return bounding_boxes, class_names, confidences


def require_runtime_files(args):
    """Fail with the build command instead of a TensorRT deserialization error."""
    missing = [
        path
        for path in (
            args.vision_engine,
            args.text_engine,
            args.decoder_engine,
            args.tokenizer,
        )
        if not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError(
            "SAM3 runtime files missing:\n  "
            + "\n  ".join(missing)
            + "\nRun: bash download_sam3_onnx.sh runtime && bash wrap_sam3_trt.sh"
        )
