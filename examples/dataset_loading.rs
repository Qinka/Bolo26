/// Example: Dataset loading
///
/// This example demonstrates how to:
/// 1. Create a YOLO dataset
/// 2. Load data from directory
/// 3. Access dataset items

use yolo26::data::{YoloDataset, BoundingBox};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("YOLO v26 Dataset Example");
    println!("========================\n");

    // 1. Create empty dataset
    let num_classes = 80;
    let dataset = YoloDataset::new(num_classes);
    println!("Created dataset for {} classes", num_classes);
    println!("Dataset size: {}\n", dataset.len());

    // 2. Example: How to load from directory
    println!("To load dataset from directory:");
    println!("  let dataset = YoloDataset::from_directory(\"path/to/data\", 80)?;");
    println!("\nExpected directory structure:");
    println!("  data/");
    println!("  ├── images/");
    println!("  │   ├── image1.jpg");
    println!("  │   └── image2.jpg");
    println!("  └── labels/");
    println!("      ├── image1.txt");
    println!("      └── image2.txt\n");

    // 3. Example: Bounding box format
    println!("Label file format (YOLO):");
    println!("  class_id center_x center_y width height");
    println!("  Example: 0 0.5 0.5 0.3 0.4\n");

    // 4. Create example bounding box
    let bbox = BoundingBox::new(0.5, 0.5, 0.3, 0.4, 0);
    println!("Example bounding box:");
    println!("  Center: ({}, {})", bbox.x, bbox.y);
    println!("  Size: {}x{}", bbox.width, bbox.height);
    println!("  Class: {}", bbox.class_id);

    Ok(())
}
