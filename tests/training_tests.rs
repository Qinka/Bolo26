use yolo26::training::{TrainingConfig, YoloLoss};

#[test]
fn test_training_config_creation() {
    let config = TrainingConfig::new();
    
    assert_eq!(config.num_epochs, 100);
    assert_eq!(config.batch_size, 16);
    assert_eq!(config.learning_rate, 0.001);
}

#[test]
fn test_training_config_builder() {
    let config = TrainingConfig::new()
        .with_epochs(200)
        .with_batch_size(32)
        .with_learning_rate(0.0001);
    
    assert_eq!(config.num_epochs, 200);
    assert_eq!(config.batch_size, 32);
    assert_eq!(config.learning_rate, 0.0001);
}

#[test]
fn test_loss_creation() {
    let _loss = YoloLoss::new();
    // If we get here without panic, the loss was created successfully
}

#[test]
fn test_loss_with_weights() {
    let _loss = YoloLoss::with_weights(0.05, 1.0, 0.5);
    // If we get here without panic, the loss was created successfully
}
