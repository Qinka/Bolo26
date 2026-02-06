use burn::prelude::*;
use super::backbone::ConvBlock;

/// Upsample layer
#[derive(Module, Debug)]
pub struct Upsample<B: Backend> {
    scale_factor: usize,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> Upsample<B> {
    pub fn new(scale_factor: usize) -> Self {
        Self {
            scale_factor,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        // Simple nearest neighbor upsampling using repeat
        let [batch, channels, height, width] = input.dims();
        
        // Reshape to add dimensions for upsampling: [B, C, H, 1, W, 1]
        let x = input.reshape([batch, channels, height, 1, width, 1]);
        
        // Repeat along the new dimensions
        let x = x.repeat(&[1, 1, 1, self.scale_factor, 1, self.scale_factor]);
        
        // Reshape back to [B, C, H*scale, W*scale]
        let new_height = height * self.scale_factor;
        let new_width = width * self.scale_factor;
        x.reshape([batch, channels, new_height, new_width])
    }
}

/// Path Aggregation Feature Pyramid Network (PAFPN)
#[derive(Module, Debug)]
pub struct PAFPN<B: Backend> {
    // Top-down pathway
    lateral_conv1: ConvBlock<B>,
    lateral_conv2: ConvBlock<B>,
    
    // Bottom-up pathway  
    downsample_conv1: ConvBlock<B>,
    downsample_conv2: ConvBlock<B>,
    
    // Feature fusion
    fpn_conv1: ConvBlock<B>,
    fpn_conv2: ConvBlock<B>,
    fpn_conv3: ConvBlock<B>,
    
    upsample: Upsample<B>,
}

impl<B: Backend> PAFPN<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            lateral_conv1: ConvBlock::new(1024, 512, 1, 1, device),
            lateral_conv2: ConvBlock::new(512, 256, 1, 1, device),
            
            downsample_conv1: ConvBlock::new(256, 256, 3, 2, device),
            downsample_conv2: ConvBlock::new(512, 512, 3, 2, device),
            
            fpn_conv1: ConvBlock::new(512, 256, 3, 1, device),
            fpn_conv2: ConvBlock::new(768, 512, 3, 1, device),
            fpn_conv3: ConvBlock::new(1536, 1024, 3, 1, device),
            
            upsample: Upsample::new(2),
        }
    }

    pub fn forward(
        &self,
        p3: Tensor<B, 4>,
        p4: Tensor<B, 4>,
        p5: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        // Top-down pathway
        let p5_lateral = self.lateral_conv1.forward(p5);
        let p5_up = self.upsample.forward(p5_lateral.clone());
        
        let p4_fused = Tensor::cat(vec![p5_up, p4], 1);
        let p4_lateral = self.lateral_conv2.forward(p4_fused);
        let p4_up = self.upsample.forward(p4_lateral.clone());
        
        let p3_fused = Tensor::cat(vec![p4_up, p3], 1);
        let p3_out = self.fpn_conv1.forward(p3_fused);
        
        // Bottom-up pathway
        let p3_down = self.downsample_conv1.forward(p3_out.clone());
        let n4 = Tensor::cat(vec![p3_down, p4_lateral], 1);
        let n4_out = self.fpn_conv2.forward(n4);
        
        let n4_down = self.downsample_conv2.forward(n4_out.clone());
        let n5 = Tensor::cat(vec![n4_down, p5_lateral], 1);
        let n5_out = self.fpn_conv3.forward(n5);
        
        (p3_out, n4_out, n5_out)
    }
}
