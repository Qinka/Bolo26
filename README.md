# Bolo26

**YOLO v26 implementation in Rust using the Burn deep learning framework**

A modern, performant implementation of YOLO (You Only Look Once) object detection using Rust and the Burn framework. This project provides training capabilities and ONNX export for edge deployment.

## Features

- 🦀 **Pure Rust Implementation**: Built with Rust for safety, performance, and reliability
- 🔥 **Burn Framework**: Leverages the modern Burn deep learning framework
- 🎯 **YOLO v26 Architecture**: State-of-the-art object detection model
- 🚀 **Multi-Backend Support**: NdArray (CPU), LibTorch (GPU), and WGPU backends
- 📦 **ONNX Export**: Export models for edge deployment and inference
- 🏋️ **Training Pipeline**: Complete training infrastructure with loss functions and optimizers

## Architecture

The YOLO v26 model consists of three main components:

1. **Backbone (CSPDarknet)**: Feature extraction network with Cross Stage Partial connections
2. **Neck (PAFPN)**: Path Aggregation Feature Pyramid Network for multi-scale features
3. **Head**: Multi-scale detection heads for object detection at different scales

## Installation

### Prerequisites

- Rust 1.70 or later
- Cargo package manager

### Build

```bash
# Clone the repository
git clone https://github.com/Qinka/Bolo26.git
cd Bolo26

# Build the project
cargo build --release
```

## Usage

### Quick Start

```bash
# Run the main application
cargo run --bin yolo26
```

### Training

```bash
# Train the model
cargo run --bin train --release
```

The training pipeline includes:
- Multi-scale detection training
- Box regression loss
- Objectness loss  
- Classification loss

### Inference

```bash
# Run inference on test data
cargo run --bin infer --release
```

### ONNX Export

```bash
# Export model to ONNX format for edge deployment
cargo run --bin export-onnx --release
```

The ONNX export allows you to:
- Deploy models on edge devices
- Run inference in non-Rust environments
- Optimize for specific hardware platforms

## Model Configuration

Default configuration (can be customized):

- **Input Size**: 640x640
- **Number of Classes**: 80 (COCO dataset)
- **Confidence Threshold**: 0.25
- **IoU Threshold**: 0.45

Example configuration:

```rust
use yolo26::model::Yolo26Config;

let config = Yolo26Config::new(80)
    .with_input_size(640)
    .with_conf_threshold(0.25)
    .with_iou_threshold(0.45);
```

## Project Structure

```
Bolo26/
├── src/
│   ├── model/           # Model architecture
│   │   ├── backbone.rs  # CSPDarknet backbone
│   │   ├── neck.rs      # PAFPN neck
│   │   ├── head.rs      # Detection heads
│   │   └── yolo.rs      # Main YOLO model
│   ├── data/            # Dataset loading and augmentation
│   ├── training/        # Training pipeline
│   ├── utils/           # Utilities (NMS, visualization)
│   ├── bin/             # Binary executables
│   └── lib.rs           # Library root
├── Cargo.toml           # Project dependencies
└── README.md            # This file
```

## Features Flags

The project supports multiple backend options:

```toml
[features]
default = ["ndarray"]
ndarray = []           # CPU backend (default)
tch = ["burn-tch"]     # LibTorch GPU backend
wgpu = ["burn-wgpu"]   # WGPU backend
```

Build with a specific backend:

```bash
# Build with LibTorch backend
cargo build --release --features tch

# Build with WGPU backend
cargo build --release --features wgpu
```

## Dataset Format

The project supports YOLO format datasets:

```
dataset/
├── images/
│   ├── image1.jpg
│   └── image2.jpg
└── labels/
    ├── image1.txt
    └── image2.txt
```

Label format (YOLO):
```
class_id center_x center_y width height
```

## Training Configuration

Customize training parameters:

```rust
use yolo26::training::TrainingConfig;

let config = TrainingConfig::new()
    .with_epochs(100)
    .with_batch_size(16)
    .with_learning_rate(0.001);
```

## API Usage

### Creating a Model

```rust
use yolo26::Yolo26;
use burn::backend::NdArray;

type Backend = NdArray;

fn main() {
    let device = Default::default();
    let num_classes = 80;
    
    let model = Yolo26::<Backend>::new(num_classes, &device);
    
    // Forward pass
    let input = /* load your image tensor */;
    let (small, medium, large) = model.forward(input);
}
```

### Post-Processing

```rust
use yolo26::utils::non_max_suppression;

let boxes = /* extract boxes from predictions */;
let scores = /* extract scores */;
let filtered_boxes = non_max_suppression(boxes, scores, 0.45);
```

## Performance

The implementation is designed for:
- Fast inference on both CPU and GPU
- Memory-efficient training
- Scalable to large datasets

## Limitations and Future Work

- [ ] Complete ONNX export implementation (waiting for full Burn support)
- [ ] Add pre-trained weights loading
- [ ] Implement data augmentation pipeline
- [ ] Add model quantization support
- [ ] Complete loss function implementation with box matching
- [ ] Add evaluation metrics (mAP)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under MIT OR Apache-2.0.

## Acknowledgments

- Built with [Burn](https://github.com/tracel-ai/burn) deep learning framework
- Inspired by the official YOLO implementations
- YOLO architecture by Ultralytics

## References

- [Burn Framework](https://github.com/tracel-ai/burn)
- [YOLO Object Detection](https://github.com/ultralytics/ultralytics)
- [ONNX](https://onnx.ai/)

