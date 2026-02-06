use yolo26::{Yolo26, training::{Trainer, TrainingConfig}};
use burn::backend::NdArray;

type Backend = NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("YOLO v26 Training");
    println!("=================\n");

    // Configuration
    let num_classes = 80; // COCO dataset
    let device = Default::default();

    // Create model
    println!("Initializing model...");
    let model = Yolo26::<Backend>::new(num_classes, &device);
    println!("Model created with {} classes", num_classes);

    // Training configuration
    let config = TrainingConfig::new()
        .with_epochs(100)
        .with_batch_size(16)
        .with_learning_rate(0.001);

    println!("\nTraining Configuration:");
    println!("  Epochs: {}", config.num_epochs);
    println!("  Batch size: {}", config.batch_size);
    println!("  Learning rate: {}", config.learning_rate);

    // Create trainer
    let mut trainer = Trainer::new(model, config);

    // Start training
    println!("\nStarting training...");
    trainer.train()?;

    println!("\nTraining completed!");

    Ok(())
}
