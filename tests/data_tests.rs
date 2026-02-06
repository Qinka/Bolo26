use yolo26::data::{YoloDataset, BoundingBox};

#[test]
fn test_dataset_creation() {
    let dataset = YoloDataset::new(80);
    assert_eq!(dataset.num_classes(), 80);
    assert_eq!(dataset.len(), 0);
    assert!(dataset.is_empty());
}

#[test]
fn test_bounding_box_creation() {
    let bbox = BoundingBox::new(0.5, 0.5, 0.3, 0.4, 0);
    
    assert_eq!(bbox.x, 0.5);
    assert_eq!(bbox.y, 0.5);
    assert_eq!(bbox.width, 0.3);
    assert_eq!(bbox.height, 0.4);
    assert_eq!(bbox.class_id, 0);
}

#[test]
fn test_bounding_box_yolo_format() {
    let bbox = BoundingBox::new(0.5, 0.5, 0.3, 0.4, 0);
    let yolo_format = bbox.to_yolo_format();
    
    assert_eq!(yolo_format, [0.5, 0.5, 0.3, 0.4]);
}

#[test]
fn test_bounding_box_from_yolo_format() {
    let bbox = BoundingBox::from_yolo_format([0.5, 0.5, 0.3, 0.4], 0);
    
    assert_eq!(bbox.x, 0.5);
    assert_eq!(bbox.y, 0.5);
    assert_eq!(bbox.width, 0.3);
    assert_eq!(bbox.height, 0.4);
    assert_eq!(bbox.class_id, 0);
}
