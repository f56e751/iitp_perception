import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
import supervision as sv
from pathlib import Path
from torchvision.ops import box_convert
import grounding_dino.groundingdino.datasets.transforms as T
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import argparse
import sys
import time
from collections import defaultdict
from typing import List, Tuple

BOX_THRESHOLD = 0.2 ## minimum to be valid bbox
TEXT_THRESHOLD = 0.4
TRANSPARANTS_TEXT_THRESHOLD = 0.7
OUTPUT_DIR =  os.path.join("tmp_results/",  f"results_{BOX_THRESHOLD}_{TEXT_THRESHOLD}")

GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "checkpoint_best.pth" #path/to/weight

TEXT_PROMPT = "786dvpteg. k3m9t8z1q. d7f2x4b6n." #transparent(786dvpteg), metal(k3m9t8z1q), cardboard(d7f2x4b6n)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    IoU between one box (4,) and many boxes (N,4), all in xyxy.
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.clip(x2 - x1, a_min=0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0, a_max=None)
    inter = inter_w * inter_h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter + 1e-9
    return inter / union


def nms_by_label(
    confidences: np.ndarray,
    input_boxes: np.ndarray,          # shape (N,4), xyxy
    labels: List[str],
    iou_thresh: float = 0.5,
    method: str = "hard",             # "hard" | "soft"
    sigma: float = 0.5,               # soft-nms parameter
    score_thresh: float = 1e-3        # drop boxes with score < score_thresh (soft-nms only)
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Per-label NMS.
    Returns:
        kept_boxes   : (M,4) float32
        kept_scores  : (M,) float32
        kept_labels  : list[str] length M
        keep_indices : (M,) int64 original indices in the input arrays
    """
    assert input_boxes.ndim == 2 and input_boxes.shape[1] == 4, "input_boxes must be (N,4) xyxy"
    assert len(confidences) == len(input_boxes) == len(labels), "length mismatch"

    if len(input_boxes) == 0:
        return (input_boxes.astype(np.float32),
                confidences.astype(np.float32),
                [],
                np.empty((0,), dtype=np.int64))

    # group by label
    groups = defaultdict(list)
    for idx, lab in enumerate(labels):
        groups[lab].append(idx)

    kept_boxes = []
    kept_scores = []
    kept_labels = []
    kept_indices = []

    for lab, idxs in groups.items():
        idxs = np.array(idxs, dtype=np.int64)
        boxes = input_boxes[idxs].astype(np.float32, copy=True)
        scores = confidences[idxs].astype(np.float32, copy=True)

        if method.lower() == "hard":
            # sort by score desc
            order = np.argsort(-scores)
            boxes = boxes[order]
            scores = scores[order]
            idxs  = idxs[order]

            keep = []
            while len(boxes) > 0:
                # take top-1
                keep.append(0)
                if len(boxes) == 1:
                    break
                ious = _iou_xyxy(boxes[0], boxes[1:])
                keep_mask = ious <= iou_thresh
                # keep only those with IoU <= thresh
                boxes = boxes[1:][keep_mask]
                scores = scores[1:][keep_mask]
                idxs   = idxs[1:][keep_mask]

            keep = np.array(keep, dtype=np.int64)
            # reconstruct kept arrays from the original "order"-sorted arrays
            order_sorted_boxes = input_boxes[np.array(groups[lab], dtype=np.int64)][np.argsort(-confidences[idxs])]
            # Actually simpler: we tracked idxs after suppression; just gather those original indices.
            # Since we mutated idxs, we need to rebuild kept using the saved indices along the loop.

            # The above reconstruction is messy; easier: re-run with a clean loop that records chosen indices.
            # Re-implement cleanly:
            boxes = input_boxes[np.array(groups[lab], dtype=np.int64)].astype(np.float32)
            scores = confidences[np.array(groups[lab], dtype=np.int64)].astype(np.float32)
            order = np.argsort(-scores)
            boxes = boxes[order]; scores = scores[order]; idxs = np.array(groups[lab], dtype=np.int64)[order]

            selected = []
            while len(order) > 0 and len(boxes) > 0:
                selected.append((boxes[0], scores[0], idxs[0]))
                if len(boxes) == 1:
                    break
                ious = _iou_xyxy(boxes[0], boxes[1:])
                keep_mask = ious <= iou_thresh
                boxes = boxes[1:][keep_mask]
                scores = scores[1:][keep_mask]
                idxs = idxs[1:][keep_mask]

            for b, s, i in selected:
                kept_boxes.append(b)
                kept_scores.append(s)
                kept_labels.append(lab)
                kept_indices.append(i)

        elif method.lower() == "soft":
            # Soft-NMS (linear). In-place updates on copies.
            # Sort by score desc to start, then iterate.
            order = np.argsort(-scores)
            boxes = boxes[order]
            scores = scores[order]
            idxs   = idxs[order]

            selected = []
            while len(boxes) > 0:
                # pick max score box
                max_idx = np.argmax(scores)
                b = boxes[max_idx].copy()
                s = scores[max_idx].copy()
                i = idxs[max_idx].copy()
                selected.append((b, s, i))

                # remove the selected
                mask = np.ones(len(boxes), dtype=bool)
                mask[max_idx] = False
                remain_boxes = boxes[mask]
                remain_scores = scores[mask]
                remain_idxs = idxs[mask]

                if len(remain_boxes) == 0:
                    break

                # decay scores of the rest
                ious = _iou_xyxy(b, remain_boxes)
                decay = np.where(ious > iou_thresh, (1 - ious), 1.0)  # linear
                remain_scores = remain_scores * np.exp(- (ious**2) / sigma) if sigma < 0 else remain_scores * decay

                # filter low scores
                keep_mask = remain_scores >= score_thresh
                boxes = remain_boxes[keep_mask]
                scores = remain_scores[keep_mask]
                idxs   = remain_idxs[keep_mask]

            for b, s, i in selected:
                kept_boxes.append(b)
                kept_scores.append(s)
                kept_labels.append(lab)
                kept_indices.append(i)

        else:
            raise ValueError("method must be 'hard' or 'soft'")

    # concatenate result from all labels and sort globally by score desc
    kept_boxes = np.asarray(kept_boxes, dtype=np.float32)
    kept_scores = np.asarray(kept_scores, dtype=np.float32)
    kept_indices = np.asarray(kept_indices, dtype=np.int64)
    # stable sort by score desc
    order = np.argsort(-kept_scores)
    kept_boxes = kept_boxes[order]
    kept_scores = kept_scores[order]
    kept_indices = kept_indices[order]
    kept_labels = [kept_labels[i] for i in order]

    return kept_boxes, kept_scores, kept_labels, kept_indices


def offset_to_bbox(bbox, offset, img_width, img_height):
    x1, y1, x2, y2 = bbox
    x1_new = max(0, x1 - offset)
    y1_new = max(0, y1 - offset)
    x2_new = min(img_width, x2 + offset)
    y2_new = min(img_height, y2 + offset)
    return [x1_new, y1_new, x2_new, y2_new]
def merge_bounding_boxes(bboxes):
    bboxes = np.array(bboxes)
    x_min = np.min(bboxes[:, 0])
    y_min = np.min(bboxes[:, 1])
    x_max = np.max(bboxes[:, 2])
    y_max = np.max(bboxes[:, 3])
    return [x_min, y_min, x_max, y_max]



def process_masks_by_size(masks, labels, threshold=300):
    processed_masks = masks.copy()
    mask_sizes = [-np.sum(mask) for mask in masks]
    sorted_indices = np.argsort(mask_sizes)
    sorted_masks = processed_masks[sorted_indices]
    labels = np.array(labels)
    sorted_labels = labels[sorted_indices]
    for i in range(len(sorted_masks)):
        if np.sum(sorted_masks[i]) < threshold:
            continue
        for j in range(i + 1, len(sorted_masks)):
            overlap_region = np.logical_and(sorted_masks[j], sorted_masks[i])
            sorted_masks[j][overlap_region] = 0
    filtered_masks = [
        mask for mask in sorted_masks if np.sum(mask) >= threshold
    ]
    filtered_labels = [
        label for mask, label in zip(sorted_masks, sorted_labels) if np.sum(mask) >= threshold
    ]
    if len(filtered_masks) > 0:
        return np.array(filtered_masks), np.array(filtered_labels)
    else:
        return np.empty((0, *masks.shape[1:]), dtype=masks.dtype), np.empty((0, *labels.shape[1:]), dtype=labels.dtype)
    
def _bbox_from_mask(mask: np.ndarray):
    """Binary mask -> [x1, y1, x2, y2] (닫힌 구간, x2/y2는 max 인덱스)"""
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()
    return [int(x1), int(y1), int(x2), int(y2)]

def process_masks_and_update_boxes(
    masks: np.ndarray,
    boxes: np.ndarray = None,
    confidences: np.ndarray = None,
    labels: np.ndarray = None,
    threshold: int = 500
):
    if masks.size == 0:
        out = np.empty((0, *masks.shape[1:]), dtype=masks.dtype)
        return out, np.empty((0,4), dtype=int), \
               (np.empty((0,)) if confidences is None else confidences[:0]), \
               (np.empty((0,), dtype=labels.dtype) if labels is not None else None)

    masks = masks.astype(bool)
    areas = np.array([int(m.sum()) for m in masks])
    sorted_idx = np.argsort(-areas)
    sorted_masks = masks[sorted_idx].copy()
    for i in range(len(sorted_masks)):
        if sorted_masks[i].sum() < threshold:
            continue
        for j in range(i + 1, len(sorted_masks)):
            if sorted_masks[j].sum() == 0:
                continue
            overlap = np.logical_and(sorted_masks[i], sorted_masks[j])
            if overlap.any():
                sorted_masks[j][overlap] = 0
    keep_mask = np.array([m.sum() >= threshold for m in sorted_masks], dtype=bool)
    final_masks = sorted_masks[keep_mask]
    kept_sorted_positions = np.nonzero(keep_mask)[0]
    kept_original_indices = sorted_idx[kept_sorted_positions]
    final_boxes = []
    for m in final_masks:
        bb = _bbox_from_mask(m)
        if bb is None:
            final_boxes.append([0,0,0,0])
        else:
            final_boxes.append(bb)
    final_boxes = np.array(final_boxes, dtype=int)
    final_conf = None
    final_labels = None
    if confidences is not None:
        final_conf = confidences[kept_original_indices]
    if labels is not None:
        final_labels = labels[kept_original_indices]

    return final_masks.astype(masks.dtype), final_boxes, final_conf, final_labels

# Functions Added by JHSong for communication # 
def transform_image(image) -> Tuple[np.array, torch.Tensor]:    # modification of grounding_dino.groundingdino.util.inference.load_model
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # image_source = Image.open(image_path).convert("RGB")
    # image = np.asarray(image_source)
    image_source = Image.fromarray(image)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

counter = 0
LAST_ANNOTATED = None  # latest annotated BGR frame; consumed by external streamers
def object_detector(model, color_np, depth_np, camera_intrinsics):
    global counter, LAST_ANNOTATED
    LAST_ANNOTATED = None
    os.makedirs(OUTPUT_DIR, exist_ok=True)
# frame_names = sorted(frame_names)
# for frame_idx in range(len(frame_names)):
    ## do not use i in this loop!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    start = time.perf_counter()
    # print(frame_names[frame_idx])
    text = TEXT_PROMPT
    # img_path = os.path.join(IMG_PATH, frame_names[frame_idx])
    # image_source, image = load_image(img_path)
    rgb_np = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
    image_source, image = transform_image(rgb_np)
    print(image.shape)
    # sam2_predictor.set_image(image_source)
    t_boxes, t_confidences, t_labels = predict(
        # model=grounding_model,
        model=model,
        image=image,
        caption=text,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        remove_combined = True
    )
    if len(t_boxes)==0:
        # print(f"No object in {frame_names[frame_idx]}!!!")
        print(f"No object in {counter}!!!")
        counter += 1
        # continue
        return [], []

    labels = []
    mask = torch.zeros_like(t_confidences)<1
    
    for i, label in enumerate(t_labels):
        if label in ["786dvpteg", "k3m9t8z1q", "d7f2x4b6n"]:
            # if label == "786dvpteg" and t_confidences[i] <0.7:
            #     labels.append("k3m9t8z1q")
            # else:
            #     labels.append(label)
            labels.append(label)
        else:
            mask[i]=False

    confidences = t_confidences[mask]
    boxes = t_boxes[mask]

    # post process
    h, w, _ = image_source.shape
    boxes_px = (boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device))
    input_boxes = box_convert(boxes=boxes_px, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
    confidences = np.asarray(confidences, dtype=float)

    input_boxes, 
    input_boxes, confidences, labels, kept_idx = nms_by_label(
        confidences=confidences,           # np.ndarray, shape (N,)
        input_boxes=input_boxes,           # np.ndarray, shape (N,4), xyxy
        labels=labels,                     # list[str], len N
        iou_thresh=0.5,
        method="soft"                      # 또는 "soft"
    )

            

    class_names = np.asarray(labels)
    class_names = np.array([s.replace('786dvpteg', 'transparent') for s in class_names])
    class_names = np.array([s.replace('d7f2x4b6n', 'cardboard') for s in class_names])
    class_names = np.array([s.replace('k3m9t8z1q', 'metal') for s in class_names])
    class_ids = np.arange(len(class_names))  # or map via your fixed label_map



    ## visulaization

    detections = sv.Detections(
        xyxy=input_boxes,          # (M,4)  <- boxes_updated
        mask=None,          # (M,H,W) <- masks after overlap removal
        class_id=class_ids     # (M,)
    )
    if confidences is not None:
        label_texts = [f"{name} {score:.2f}" for name, score in zip(class_names, confidences)]

    print(label_texts)

    img = image_source
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    mask_annotator = sv.MaskAnnotator(opacity=0.4)
    box_annotator  = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=1.0)

    annotated = img.copy()
    annotated = mask_annotator.annotate(scene=annotated, detections=detections)
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=label_texts)
    # out_path = os.path.join(OUTPUT_DIR, frame_names[frame_idx])
    out_path = os.path.join(OUTPUT_DIR, f'{counter}.jpg')
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    cv2.imwrite(out_path, annotated)
    LAST_ANNOTATED = annotated
    elapsed = time.perf_counter() - start
    # print(f"{frame_idx}th image is finished. total time: {elapsed:.3f} s")
    print(f"{counter}th image is finished. total time: {elapsed:.3f} s")
    counter += 1    # analgous to frame_idx in the original code

    # Changing 2D information to 3D
    positions = []
    for input_box in input_boxes:
        u, v = (input_box[0] + input_box[2]) / 2, (input_box[1] + input_box[3]) / 2
        depth = depth_np[int(v), int(u)] / 1000.0  # Assuming mm to m
        if depth <= 0.1 or depth >= 5.0:
            pass    # raise exception?

        fx, fy, cx, cy = camera_intrinsics
        X, Y, Z = (u - cx) / fx * depth, (v - cy) / fy * depth, depth
        positions.append((X, Y, Z))
    return positions, class_names.tolist()


def run_batch(input_dir):
    images_dir = os.path.join(input_dir, "images")
    frame_names = [
        p for p in os.listdir(images_dir)
        if os.path.splitext(p)[-1] in [".JPG", ".jpeg", ".jpg", ".JPEG", ".png"]
    ]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )

    frame_names = sorted(frame_names)
    for frame_idx in range(len(frame_names)):
        ## do not use i in this loop!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        start = time.perf_counter()
        print(frame_names[frame_idx])
        text = TEXT_PROMPT
        img_path = os.path.join(images_dir, frame_names[frame_idx])
        image_source, image = load_image(img_path)
        print(image.shape)
        # sam2_predictor.set_image(image_source)
        t_boxes, t_confidences, t_labels = predict(
            model=grounding_model,
            image=image,
            caption=text,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            remove_combined = True
        )
        if len(t_boxes)==0:
            print(f"No object in {frame_names[frame_idx]}!!!")
            continue

        labels = []
        mask = torch.zeros_like(t_confidences)<1
        
        for i, label in enumerate(t_labels):
            if label in ["786dvpteg", "k3m9t8z1q", "d7f2x4b6n"]:
                # if label == "786dvpteg" and t_confidences[i] <0.7:
                #     labels.append("k3m9t8z1q")
                # else:
                #     labels.append(label)
                labels.append(label)
            else:
                mask[i]=False

        confidences = t_confidences[mask]
        boxes = t_boxes[mask]

        # post process
        h, w, _ = image_source.shape
        boxes_px = (boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device))
        input_boxes = box_convert(boxes=boxes_px, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        confidences = np.asarray(confidences, dtype=float)

        input_boxes, 
        input_boxes, confidences, labels, kept_idx = nms_by_label(
            confidences=confidences,           # np.ndarray, shape (N,)
            input_boxes=input_boxes,           # np.ndarray, shape (N,4), xyxy
            labels=labels,                     # list[str], len N
            iou_thresh=0.5,
            method="soft"                      # 또는 "soft"
        )

                

        class_names = np.asarray(labels)
        class_names = np.array([s.replace('786dvpteg', 'transparent') for s in class_names])
        class_names = np.array([s.replace('d7f2x4b6n', 'cardboard') for s in class_names])
        class_names = np.array([s.replace('k3m9t8z1q', 'metal') for s in class_names])
        class_ids = np.arange(len(class_names))  # or map via your fixed label_map



        ## visulaization

        detections = sv.Detections(
            xyxy=input_boxes,          # (M,4)  <- boxes_updated
            mask=None,          # (M,H,W) <- masks after overlap removal
            class_id=class_ids     # (M,)
        )
        if confidences is not None:
            label_texts = [f"{name} {score:.2f}" for name, score in zip(class_names, confidences)]

        print(label_texts)

        img = image_source
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        mask_annotator = sv.MaskAnnotator(opacity=0.4)
        box_annotator  = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=1.0)

        annotated = img.copy()
        annotated = mask_annotator.annotate(scene=annotated, detections=detections)
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=label_texts)
        out_path = os.path.join(OUTPUT_DIR, frame_names[frame_idx])
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        cv2.imwrite(out_path, annotated)
        elapsed = time.perf_counter() - start
        print(f"{frame_idx}th image is finished. total time: {elapsed:.3f} s")


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch detection on a folder of images (<input-dir>/images/)."
    )
    p.add_argument(
        "-i", "--input_dir", required=True,
        help="Folder containing an images/ subdirectory.",
    )
    return p.parse_args()


def main():
    run_batch(parse_args().input_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())