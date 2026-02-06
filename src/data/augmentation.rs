/// Data augmentation for YOLO training
pub struct Augmentation {
    pub horizontal_flip: bool,
    pub random_crop: bool,
    pub color_jitter: bool,
}

impl Default for Augmentation {
    fn default() -> Self {
        Self {
            horizontal_flip: true,
            random_crop: false,
            color_jitter: true,
        }
    }
}

impl Augmentation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_horizontal_flip(mut self, enabled: bool) -> Self {
        self.horizontal_flip = enabled;
        self
    }

    pub fn with_random_crop(mut self, enabled: bool) -> Self {
        self.random_crop = enabled;
        self
    }

    pub fn with_color_jitter(mut self, enabled: bool) -> Self {
        self.color_jitter = enabled;
        self
    }
}
