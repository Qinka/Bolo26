use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
};
use burn::prelude::*;

/// Convolutional block with BatchNorm and activation
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B, 2>,
    activation: Relu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, device: &B::Device) -> Self {
        let padding = (kernel_size - 1) / 2;
        let conv = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(padding, padding))
            .with_bias(false)
            .init(device);
        let bn = BatchNormConfig::new(out_channels).init(device);
        let activation = Relu::new();

        Self { conv, bn, activation }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.bn.forward(x);
        self.activation.forward(x)
    }
}

/// CSP (Cross Stage Partial) Block
#[derive(Module, Debug)]
pub struct CSPBlock<B: Backend> {
    conv1: ConvBlock<B>,
    conv2: ConvBlock<B>,
    conv3: ConvBlock<B>,
    conv4: ConvBlock<B>,
}

impl<B: Backend> CSPBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, _num_blocks: usize, device: &B::Device) -> Self {
        let hidden = out_channels / 2;
        
        Self {
            conv1: ConvBlock::new(in_channels, hidden, 1, 1, device),
            conv2: ConvBlock::new(in_channels, hidden, 1, 1, device),
            conv3: ConvBlock::new(hidden, hidden, 3, 1, device),
            conv4: ConvBlock::new(hidden * 2, out_channels, 1, 1, device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x1 = self.conv1.forward(input.clone());
        let x2 = self.conv2.forward(input);
        let x2 = self.conv3.forward(x2);
        
        // Concatenate along channel dimension
        let x = Tensor::cat(vec![x1, x2], 1);
        self.conv4.forward(x)
    }
}

/// CSPDarknet Backbone
#[derive(Module, Debug)]
pub struct CSPDarknet<B: Backend> {
    stem: ConvBlock<B>,
    stage1: CSPBlock<B>,
    stage2: CSPBlock<B>,
    stage3: CSPBlock<B>,
    stage4: CSPBlock<B>,
    stage5: CSPBlock<B>,
}

impl<B: Backend> CSPDarknet<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            stem: ConvBlock::new(3, 32, 3, 1, device),
            stage1: CSPBlock::new(32, 64, 1, device),
            stage2: CSPBlock::new(64, 128, 2, device),
            stage3: CSPBlock::new(128, 256, 8, device),
            stage4: CSPBlock::new(256, 512, 8, device),
            stage5: CSPBlock::new(512, 1024, 4, device),
        }
    }

    /// Forward pass returning multi-scale features
    pub fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let x = self.stem.forward(input);
        let x = self.stage1.forward(x);
        let x = self.stage2.forward(x);
        let p3 = self.stage3.forward(x);
        let p4 = self.stage4.forward(p3.clone());
        let p5 = self.stage5.forward(p4.clone());

        (p3, p4, p5)
    }
}
