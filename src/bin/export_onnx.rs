use yolo26::{Yolo26, model::Yolo26Config};
use burn::backend::NdArray;

type Backend = NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("YOLO v26 ONNX Export");
    println!("====================\n");

    // Configuration
    let config = Yolo26Config::default();
    let device = Default::default();

    // Create model
    println!("Creating model...");
    let _model = Yolo26::<Backend>::new(config.num_classes, &device);
    println!("Model created with {} classes", config.num_classes);

    let output_path = "yolo26.onnx";
    
    println!("\nExporting to ONNX format...");
    println!("Output path: {}", output_path);
    
    // Note: Burn's ONNX export is still in development
    // This is a placeholder for the actual export functionality
    println!("\nONNX export functionality:");
    println!("  - Model architecture: YOLO v26");
    println!("  - Input size: {}x{}", config.input_size, config.input_size);
    println!("  - Number of classes: {}", config.num_classes);
    println!("  - Output: Multi-scale detection (3 scales)");
    
    println!("\nNote: Full ONNX export support requires burn-import crate integration.");
    println!("The model structure is ready for export once Burn adds full ONNX support.");

    Ok(())
}
