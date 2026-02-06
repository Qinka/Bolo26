use yolo26::{Yolo26, model::Yolo26Config};
use burn::backend::NdArray;

type Backend = NdArray;

#[test]
fn test_model_creation() {
    let device = Default::default();
    let num_classes = 80;
    
    let _model = Yolo26::<Backend>::new(num_classes, &device);
    // If we get here without panic, the model was created successfully
}

#[test]
fn test_config_creation() {
    let config = Yolo26Config::new(80)
        .with_input_size(640)
        .with_conf_threshold(0.25)
        .with_iou_threshold(0.45);
    
    assert_eq!(config.num_classes, 80);
    assert_eq!(config.input_size, 640);
    assert_eq!(config.conf_threshold, 0.25);
    assert_eq!(config.iou_threshold, 0.45);
}

#[test]
fn test_default_config() {
    let config = Yolo26Config::default();
    
    assert_eq!(config.num_classes, 80);
    assert_eq!(config.input_size, 640);
}
