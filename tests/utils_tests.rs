use yolo26::utils::non_max_suppression;
use yolo26::data::BoundingBox;

#[test]
fn test_nms_empty() {
    let boxes: Vec<BoundingBox> = vec![];
    let scores: Vec<f32> = vec![];
    
    let result = non_max_suppression(boxes, scores, 0.45);
    assert!(result.is_empty());
}

#[test]
fn test_nms_single_box() {
    let boxes = vec![BoundingBox::new(0.5, 0.5, 0.3, 0.4, 0)];
    let scores = vec![0.9];
    
    let result = non_max_suppression(boxes, scores, 0.45);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_nms_multiple_boxes() {
    let boxes = vec![
        BoundingBox::new(0.5, 0.5, 0.3, 0.4, 0),
        BoundingBox::new(0.6, 0.6, 0.3, 0.4, 0),
        BoundingBox::new(0.1, 0.1, 0.2, 0.2, 0),
    ];
    let scores = vec![0.9, 0.8, 0.7];
    
    let result = non_max_suppression(boxes, scores, 0.45);
    // Should keep at least one box
    assert!(!result.is_empty());
}
