use yolo26::{Yolo26, model::Yolo26Config};
use burn::backend::NdArray;
use burn::prelude::*;

type Backend = NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("YOLO v26 Inference");
    println!("==================\n");

    // Configuration
    let config = Yolo26Config::default();
    let device = Default::default();

    // Create model
    println!("Loading model...");
    let model = Yolo26::<Backend>::new(config.num_classes, &device);
    println!("Model loaded with {} classes", config.num_classes);

    // Create dummy input for testing
    let batch_size = 1;
    let input = Tensor::<Backend, 4>::zeros(
        [batch_size, 3, config.input_size, config.input_size],
        &device,
    );

    println!("\nRunning inference...");
    let (small, medium, large) = model.forward(input);

    println!("Output shapes:");
    println!("  Small scale: {:?}", small.dims());
    println!("  Medium scale: {:?}", medium.dims());
    println!("  Large scale: {:?}", large.dims());

    println!("\nInference completed!");

    Ok(())
}
