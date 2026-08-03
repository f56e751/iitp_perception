"""Lightweight bounding-box projection helpers (no ML/camera dependencies)."""


def project_bounding_boxes(input_boxes, project, depth):
    """Project xyxy pixel boxes to clockwise belt-plane corner coordinates."""
    return [
        [
            project(x1, y1, depth),
            project(x2, y1, depth),
            project(x2, y2, depth),
            project(x1, y2, depth),
        ]
        for x1, y1, x2, y2 in input_boxes
    ]
