use crate::data::BoundingBox;

/// Compute IoU (Intersection over Union) between two boxes
pub fn compute_iou(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    let x1_min = box1.x - box1.width / 2.0;
    let y1_min = box1.y - box1.height / 2.0;
    let x1_max = box1.x + box1.width / 2.0;
    let y1_max = box1.y + box1.height / 2.0;

    let x2_min = box2.x - box2.width / 2.0;
    let y2_min = box2.y - box2.height / 2.0;
    let x2_max = box2.x + box2.width / 2.0;
    let y2_max = box2.y + box2.height / 2.0;

    let inter_xmin = x1_min.max(x2_min);
    let inter_ymin = y1_min.max(y2_min);
    let inter_xmax = x1_max.min(x2_max);
    let inter_ymax = y1_max.min(y2_max);

    let inter_width = (inter_xmax - inter_xmin).max(0.0);
    let inter_height = (inter_ymax - inter_ymin).max(0.0);
    let inter_area = inter_width * inter_height;

    let box1_area = box1.width * box1.height;
    let box2_area = box2.width * box2.height;
    let union_area = box1_area + box2_area - inter_area;

    if union_area > 0.0 {
        inter_area / union_area
    } else {
        0.0
    }
}

/// Non-Maximum Suppression
pub fn non_max_suppression(
    boxes: Vec<BoundingBox>,
    scores: Vec<f32>,
    iou_threshold: f32,
) -> Vec<BoundingBox> {
    if boxes.is_empty() {
        return vec![];
    }

    // Sort boxes by score in descending order
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; boxes.len()];

    for &idx in &indices {
        if suppressed[idx] {
            continue;
        }

        keep.push(boxes[idx].clone());

        for &other_idx in &indices {
            if other_idx == idx || suppressed[other_idx] {
                continue;
            }

            let iou = compute_iou(&boxes[idx], &boxes[other_idx]);
            if iou > iou_threshold {
                suppressed[other_idx] = true;
            }
        }
    }

    keep
}
