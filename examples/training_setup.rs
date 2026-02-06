/// Example: Training configuration and setup
///
/// This example demonstrates how to:
/// 1. Set up training configuration
/// 2. Create a trainer
/// 3. Configure training parameters

use yolo26::{Yolo26, training::{TrainingConfig, Trainer}};
use burn::backend::NdArray;

type Backend = NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("YOLO v26 Training Example");
    println!("=========================\n");

    // 1. Create model
    let device = Default::default();
    let num_classes = 80;
    let model = Yolo26::<Backend>::new(num_classes, &device);
    println!("Model created with {} classes\n", num_classes);

    // 2. Configure training
    let config = TrainingConfig::new()
        .with_epochs(100)
        .with_batch_size(16)
        .with_learning_rate(0.001);

    println!("Training configuration:");
    println!("  Epochs: {}", config.num_epochs);
    println!("  Batch size: {}", config.batch_size);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Weight decay: {}", config.weight_decay);
    println!("  Save directory: {}\n", config.save_dir);

    // 3. Create trainer
    let trainer = Trainer::new(model, config);
    println!("Trainer created successfully!");

    // Note: Actual training would require:
    // - Dataset preparation
    // - Data loaders
    // - Training loop implementation
    println!("\nTo start training:");
    println!("  1. Prepare your dataset in YOLO format");
    println!("  2. Load dataset using YoloDataset");
    println!("  3. Call trainer.train()");

    Ok(())
}
