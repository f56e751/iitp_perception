"""Evaluate the IITP detection module on a folder of pre-captured images and
emit per-class scores.

Reads <input-dir>/images/*.{jpg,jpeg,png} and writes:
  <input-dir>/annotated/<name>.jpg  — annotated detection visualizations
  <input-dir>/scores.csv            — one row per kept detection with
                                       score_transparent, score_metal,
                                       score_cardboard (max sigmoid logit
                                       over each class's token span)

The detection pipeline mirrors iitp_object_detector.py's CLI path
(same TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, nms_by_label) but adds a
local predict_with_class_scores() so the (n_boxes, 256) raw logits are
retained for per-class scoring instead of being collapsed to the top-1.
"""

import argparse
import bisect
import csv
import sys
import time
from pathlib import Path

# Allow `python perception_eval/eval_detector.py` from the project root:
# put the project root on sys.path so project-level imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import supervision as sv
import torch
from torchvision.ops import box_convert

from grounding_dino.groundingdino.util.inference import (
    load_image,
    load_model,
    preprocess_caption,
)
from grounding_dino.groundingdino.util.utils import get_phrases_from_posmap

# iitp_object_detector.py runs argparse at module load and, when it sees
# --input-dir, attempts os.listdir on a derived path — which crashes during
# import because the path concat there is broken for our layout. Suppress
# argv just for this one import so the module loads cleanly.
_argv_backup = sys.argv[:]
sys.argv = sys.argv[:1]
try:
    from iitp_object_detector import (
        BOX_THRESHOLD,
        GROUNDING_DINO_CHECKPOINT,
        GROUNDING_DINO_CONFIG,
        TEXT_PROMPT,
        TEXT_THRESHOLD,
        nms_by_label,
    )
finally:
    sys.argv = _argv_backup


# Order here MUST match the order of phrases in TEXT_PROMPT.
CLASS_CODES = ["786dvpteg", "k3m9t8z1q", "d7f2x4b6n"]
CLASS_NAMES = ["transparent", "metal", "cardboard"]
CODE_TO_NAME = dict(zip(CLASS_CODES, CLASS_NAMES))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i",
        "--input-dir",
        default="tmp_results/perception_eval_260520",
        help="Folder containing images/ subdirectory.",
    )
    return p.parse_args()


def predict_with_class_scores(model, image, caption, box_threshold, device):
    """Adapted from grounding_dino.util.inference.predict, but also returns
    the raw (n_kept, 256) sigmoid logits and the prompt separator indices so
    callers can score each box against every class span.
    """
    caption = preprocess_caption(caption=caption)
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]
    prediction_boxes = outputs["pred_boxes"].cpu()[0]

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]
    boxes = prediction_boxes[mask]

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    # [CLS]=101, [SEP]=102, "."=1012
    sep_idx = [
        i
        for i, tid in enumerate(tokenized["input_ids"])
        if tid in (101, 102, 1012)
    ]

    return boxes, logits, sep_idx, tokenized, tokenizer


def class_spans(sep_idx, n_classes):
    """Return list of (left_inclusive, right_exclusive) token index ranges,
    one per class, in the order they appear in the prompt.
    """
    spans = []
    for c in range(n_classes):
        spans.append((sep_idx[c] + 1, sep_idx[c + 1]))
    return spans


def per_class_scores(logits_row, spans):
    """For a single box's (256,) logit row, return per-class max-pooled score."""
    out = []
    for left, right in spans:
        if right <= left:
            out.append(0.0)
            continue
        out.append(float(logits_row[left:right].max().item()))
    return out


def phrase_for_box(logit_row, sep_idx, text_threshold, tokenized, tokenizer):
    """Reproduce the remove_combined=True phrase extraction from predict()."""
    max_idx = int(logit_row.argmax())
    insert_idx = bisect.bisect_left(sep_idx, max_idx)
    right_idx = sep_idx[insert_idx]
    left_idx = sep_idx[insert_idx - 1]
    return (
        get_phrases_from_posmap(
            logit_row > text_threshold, tokenized, tokenizer, left_idx, right_idx
        ).replace(".", "")
    )


def list_images(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sorted(p for p in images_dir.iterdir() if p.suffix in exts)


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir)
    images_dir = input_dir / "images"
    if not images_dir.is_dir():
        print(f"ERROR: {images_dir} does not exist.", file=sys.stderr)
        return 1

    annotated_dir = input_dir / "annotated"
    annotated_dir.mkdir(exist_ok=True)
    csv_path = input_dir / "scores.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading model on {device}...", flush=True)
    model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=device,
    )

    image_paths = list_images(images_dir)
    print(f"found {len(image_paths)} images in {images_dir}", flush=True)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=1.0)

    with csv_path.open("w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "image",
                "box_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "score_transparent",
                "score_metal",
                "score_cardboard",
                "top_score",
                "predicted_class",
            ]
        )

        for img_path in image_paths:
            start = time.perf_counter()
            image_source, image = load_image(str(img_path))

            boxes, logits, sep_idx, tokenized, tokenizer = predict_with_class_scores(
                model=model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                device=device,
            )

            if len(boxes) == 0:
                print(f"{img_path.name}: no boxes above threshold", flush=True)
                continue

            spans = class_spans(sep_idx, len(CLASS_CODES))

            # Resolve each box's phrase (the existing pipeline filters by phrase
            # membership in CLASS_CODES). Mirror that filter so the CSV reflects
            # the same kept set as iitp_object_detector.py.
            phrases = [
                phrase_for_box(row, sep_idx, TEXT_THRESHOLD, tokenized, tokenizer)
                for row in logits
            ]
            keep_mask = np.array([p in CLASS_CODES for p in phrases], dtype=bool)
            if not keep_mask.any():
                print(f"{img_path.name}: no in-vocab detections", flush=True)
                continue

            boxes_kept = boxes[keep_mask]
            logits_kept = logits[keep_mask]
            phrases_kept = [p for p, k in zip(phrases, keep_mask) if k]

            h, w, _ = image_source.shape
            boxes_px = boxes_kept * torch.tensor(
                [w, h, w, h], dtype=boxes_kept.dtype, device=boxes_kept.device
            )
            input_boxes = (
                box_convert(boxes=boxes_px, in_fmt="cxcywh", out_fmt="xyxy")
                .cpu()
                .numpy()
            )
            top_scores = logits_kept.max(dim=1)[0].cpu().numpy().astype(float)

            # Per-class scores aligned to logits_kept rows (pre-NMS).
            per_class = np.array(
                [per_class_scores(row, spans) for row in logits_kept],
                dtype=float,
            )

            input_boxes, kept_top_scores, kept_codes, kept_idx = nms_by_label(
                confidences=top_scores,
                input_boxes=input_boxes,
                labels=phrases_kept,
                iou_thresh=0.5,
                method="soft",
            )
            per_class_final = per_class[kept_idx]
            class_names = [CODE_TO_NAME[c] for c in kept_codes]

            detections = sv.Detections(
                xyxy=input_boxes,
                class_id=np.arange(len(class_names)),
            )
            label_texts = [
                f"{name} {score:.2f}"
                for name, score in zip(class_names, kept_top_scores)
            ]

            annotated = image_source.copy()
            annotated = box_annotator.annotate(scene=annotated, detections=detections)
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=label_texts
            )
            annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(annotated_dir / img_path.name), annotated)

            for box_id, (box, scores, top, name) in enumerate(
                zip(input_boxes, per_class_final, kept_top_scores, class_names)
            ):
                x1, y1, x2, y2 = (float(v) for v in box)
                writer.writerow(
                    [
                        img_path.name,
                        box_id,
                        f"{x1:.1f}",
                        f"{y1:.1f}",
                        f"{x2:.1f}",
                        f"{y2:.1f}",
                        f"{scores[0]:.4f}",  # transparent (prompt index 0)
                        f"{scores[1]:.4f}",  # metal
                        f"{scores[2]:.4f}",  # cardboard
                        f"{float(top):.4f}",
                        name,
                    ]
                )

            elapsed = time.perf_counter() - start
            print(
                f"{img_path.name}: {len(class_names)} dets in {elapsed:.2f}s",
                flush=True,
            )

    print(f"wrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
