use burn::prelude::*;
use super::{CSPDarknet, PAFPN};
use super::head::MultiScaleDetectionHead;

/// YOLO v26 Model
#[derive(Module, Debug)]
pub struct Yolo26<B: Backend> {
    backbone: CSPDarknet<B>,
    neck: PAFPN<B>,
    head: MultiScaleDetectionHead<B>,
    num_classes: usize,
}

impl<B: Backend> Yolo26<B> {
    /// Create a new YOLO v26 model
    pub fn new(num_classes: usize, device: &B::Device) -> Self {
        Self {
            backbone: CSPDarknet::new(device),
            neck: PAFPN::new(device),
            head: MultiScaleDetectionHead::new(num_classes, device),
            num_classes,
        }
    }

    /// Forward pass through the model
    pub fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        // Extract features at multiple scales
        let (p3, p4, p5) = self.backbone.forward(input);
        
        // Feature pyramid network
        let (n3, n4, n5) = self.neck.forward(p3, p4, p5);
        
        // Detection heads
        self.head.forward(n3, n4, n5)
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

/// Configuration for YOLO v26
#[derive(Debug, Clone)]
pub struct Yolo26Config {
    pub num_classes: usize,
    pub input_size: usize,
    pub conf_threshold: f32,
    pub iou_threshold: f32,
}

impl Default for Yolo26Config {
    fn default() -> Self {
        Self {
            num_classes: 80, // COCO dataset default
            input_size: 640,
            conf_threshold: 0.25,
            iou_threshold: 0.45,
        }
    }
}

impl Yolo26Config {
    pub fn new(num_classes: usize) -> Self {
        Self {
            num_classes,
            ..Default::default()
        }
    }

    pub fn with_input_size(mut self, size: usize) -> Self {
        self.input_size = size;
        self
    }

    pub fn with_conf_threshold(mut self, threshold: f32) -> Self {
        self.conf_threshold = threshold;
        self
    }

    pub fn with_iou_threshold(mut self, threshold: f32) -> Self {
        self.iou_threshold = threshold;
        self
    }
}
