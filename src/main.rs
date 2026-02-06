use yolo26::{Yolo26, model::Yolo26Config};
use burn::backend::NdArray;

type Backend = NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("YOLO v26 - Rust Burn Implementation");
    println!("====================================\n");

    let config = Yolo26Config::default();
    let device = Default::default();

    println!("Creating YOLO v26 model...");
    let _model = Yolo26::<Backend>::new(config.num_classes, &device);

    println!("Model initialized successfully!");
    println!("  Classes: {}", config.num_classes);
    println!("  Input size: {}x{}", config.input_size, config.input_size);
    println!("\nAvailable commands:");
    println!("  cargo run --bin train       - Train the model");
    println!("  cargo run --bin infer       - Run inference");
    println!("  cargo run --bin export-onnx - Export to ONNX");

    Ok(())
}
