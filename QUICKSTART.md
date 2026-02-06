# Quick Start Guide

## Installation

```bash
git clone https://github.com/Qinka/Bolo26.git
cd Bolo26
cargo build --release
```

## Running the Application

### Main Application
```bash
cargo run --bin yolo26
```

### Training
```bash
cargo run --bin train --release
```

### Inference
```bash
cargo run --bin infer --release
```

### ONNX Export
```bash
cargo run --bin export-onnx --release
```

## Running Examples

```bash
# Basic usage example
cargo run --example basic_usage

# Dataset loading example
cargo run --example dataset_loading

# Training setup example
cargo run --example training_setup
```

## Running Tests

```bash
# Run all tests
cargo test

# Run specific test suite
cargo test --test data_tests
cargo test --test model_tests
cargo test --test training_tests
cargo test --test utils_tests
```

## Building for Different Backends

### CPU (Default - NdArray)
```bash
cargo build --release
```

### GPU (LibTorch)
```bash
cargo build --release --features tch
```

### WGPU
```bash
cargo build --release --features wgpu
```

## Common Commands

### Check code
```bash
cargo check
```

### Format code
```bash
cargo fmt
```

### Run clippy linter
```bash
cargo clippy
```

### Build documentation
```bash
cargo doc --open
```

## Project Structure

```
Bolo26/
├── src/
│   ├── model/          # YOLO model architecture
│   ├── data/           # Dataset handling
│   ├── training/       # Training infrastructure
│   ├── utils/          # Utilities (NMS, etc.)
│   └── bin/            # Executable binaries
├── examples/           # Usage examples
├── tests/              # Test suites
├── Cargo.toml          # Dependencies
├── README.md           # Main documentation
└── ARCHITECTURE.md     # Architecture details
```

## Quick API Reference

### Creating a Model

```rust
use yolo26::{Yolo26, model::Yolo26Config};
use burn::backend::NdArray;

type Backend = NdArray;

let device = Default::default();
let config = Yolo26Config::new(80);
let model = Yolo26::<Backend>::new(config.num_classes, &device);
```

### Loading a Dataset

```rust
use yolo26::data::YoloDataset;

let dataset = YoloDataset::from_directory("path/to/data", 80)?;
```

### Training

```rust
use yolo26::training::{Trainer, TrainingConfig};

let config = TrainingConfig::new()
    .with_epochs(100)
    .with_batch_size(16);
    
let trainer = Trainer::new(model, config);
trainer.train()?;
```

## Troubleshooting

### Build Issues

If you encounter build errors:
```bash
cargo clean
cargo build
```

### Slow Inference

Inference can be slow on CPU. For faster performance:
- Build with `--release` flag
- Use GPU backend with `--features tch`

### Memory Issues

If you run out of memory:
- Reduce batch size in training config
- Use smaller input size
- Build with `--release` for better memory management

## Next Steps

1. Prepare your dataset in YOLO format
2. Configure training parameters
3. Train the model
4. Export to ONNX for deployment
5. Deploy on edge devices

## Resources

- [Burn Documentation](https://burn.dev)
- [YOLO Documentation](https://docs.ultralytics.com)
- [Project README](README.md)
- [Architecture Guide](ARCHITECTURE.md)
