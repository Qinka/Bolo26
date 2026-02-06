use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
};
use burn::prelude::*;
use super::backbone::ConvBlock;

/// Detection head for YOLO
#[derive(Module, Debug)]
pub struct DetectionHead<B: Backend> {
    conv1: ConvBlock<B>,
    conv2: ConvBlock<B>,
    pred_conv: Conv2d<B>,
    num_classes: usize,
}

impl<B: Backend> DetectionHead<B> {
    pub fn new(in_channels: usize, num_classes: usize, device: &B::Device) -> Self {
        let hidden = 256;
        // Output channels: (4 bbox coords + 1 objectness + num_classes) * num_anchors
        let num_anchors = 3;
        let out_channels = num_anchors * (5 + num_classes);
        
        Self {
            conv1: ConvBlock::new(in_channels, hidden, 3, 1, device),
            conv2: ConvBlock::new(hidden, hidden, 3, 1, device),
            pred_conv: Conv2dConfig::new([hidden, out_channels], [1, 1])
                .with_bias(true)
                .init(device),
            num_classes,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(input);
        let x = self.conv2.forward(x);
        self.pred_conv.forward(x)
    }
}

/// Multi-scale detection head
#[derive(Module, Debug)]
pub struct MultiScaleDetectionHead<B: Backend> {
    head_small: DetectionHead<B>,
    head_medium: DetectionHead<B>,
    head_large: DetectionHead<B>,
}

impl<B: Backend> MultiScaleDetectionHead<B> {
    pub fn new(num_classes: usize, device: &B::Device) -> Self {
        Self {
            head_small: DetectionHead::new(256, num_classes, device),
            head_medium: DetectionHead::new(512, num_classes, device),
            head_large: DetectionHead::new(1024, num_classes, device),
        }
    }

    pub fn forward(
        &self,
        p3: Tensor<B, 4>,
        p4: Tensor<B, 4>,
        p5: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let small = self.head_small.forward(p3);
        let medium = self.head_medium.forward(p4);
        let large = self.head_large.forward(p5);
        
        (small, medium, large)
    }
}
