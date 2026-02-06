/// Example: Creating and using a YOLO v26 model
///
/// This example demonstrates how to:
/// 1. Create a YOLO v26 model
/// 2. Configure the model
/// 3. Run forward pass
/// 4. Process outputs

use yolo26::{Yolo26, model::Yolo26Config};
use burn::backend::NdArray;
use burn::prelude::*;

type Backend = NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("YOLO v26 Basic Example");
    println!("======================\n");

    // 1. Create configuration
    let config = Yolo26Config::new(80) // 80 classes (COCO)
        .with_input_size(640)
        .with_conf_threshold(0.25)
        .with_iou_threshold(0.45);

    println!("Model configuration:");
    println!("  Classes: {}", config.num_classes);
    println!("  Input size: {}x{}", config.input_size, config.input_size);
    println!("  Confidence threshold: {}", config.conf_threshold);
    println!("  IoU threshold: {}\n", config.iou_threshold);

    // 2. Create model
    let device = Default::default();
    let model = Yolo26::<Backend>::new(config.num_classes, &device);
    println!("Model created successfully!\n");

    // 3. Create dummy input (batch_size=1, channels=3, height=640, width=640)
    let batch_size = 1;
    let input = Tensor::<Backend, 4>::zeros(
        [batch_size, 3, config.input_size, config.input_size],
        &device,
    );
    println!("Input shape: {:?}\n", input.dims());

    // 4. Forward pass
    println!("Running forward pass...");
    let (small, medium, large) = model.forward(input);

    // 5. Print output shapes
    println!("\nOutput shapes:");
    println!("  Small scale: {:?}", small.dims());
    println!("  Medium scale: {:?}", medium.dims());
    println!("  Large scale: {:?}", large.dims());

    println!("\n✓ Example completed successfully!");

    Ok(())
}
