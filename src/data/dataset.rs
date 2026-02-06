use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Bounding box representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub class_id: usize,
}

impl BoundingBox {
    pub fn new(x: f32, y: f32, width: f32, height: f32, class_id: usize) -> Self {
        Self {
            x,
            y,
            width,
            height,
            class_id,
        }
    }

    /// Convert to YOLO format (center_x, center_y, width, height)
    pub fn to_yolo_format(&self) -> [f32; 4] {
        [self.x, self.y, self.width, self.height]
    }

    /// Create from YOLO format
    pub fn from_yolo_format(coords: [f32; 4], class_id: usize) -> Self {
        Self {
            x: coords[0],
            y: coords[1],
            width: coords[2],
            height: coords[3],
            class_id,
        }
    }
}

/// Single item in YOLO dataset
#[derive(Debug, Clone)]
pub struct YoloItem {
    pub image_path: PathBuf,
    pub boxes: Vec<BoundingBox>,
}

/// YOLO Dataset
pub struct YoloDataset {
    items: Vec<YoloItem>,
    num_classes: usize,
}

impl YoloDataset {
    pub fn new(num_classes: usize) -> Self {
        Self {
            items: Vec::new(),
            num_classes,
        }
    }

    /// Load dataset from a directory with YOLO format annotations
    pub fn from_directory<P: AsRef<Path>>(
        data_dir: P,
        num_classes: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut dataset = Self::new(num_classes);
        
        let images_dir = data_dir.as_ref().join("images");
        let labels_dir = data_dir.as_ref().join("labels");

        if !images_dir.exists() || !labels_dir.exists() {
            return Err("Images or labels directory not found".into());
        }

        // Read image files
        for entry in std::fs::read_dir(images_dir)? {
            let entry = entry?;
            let image_path = entry.path();
            
            if let Some(ext) = image_path.extension() {
                if ext == "jpg" || ext == "jpeg" || ext == "png" {
                    // Find corresponding label file
                    let stem = image_path.file_stem().unwrap();
                    let label_path = labels_dir.join(stem).with_extension("txt");
                    
                    if label_path.exists() {
                        let boxes = Self::load_boxes(&label_path)?;
                        dataset.items.push(YoloItem { image_path, boxes });
                    }
                }
            }
        }

        Ok(dataset)
    }

    fn load_boxes(label_path: &Path) -> Result<Vec<BoundingBox>, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(label_path)?;
        let mut boxes = Vec::new();

        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 5 {
                let class_id: usize = parts[0].parse()?;
                let x: f32 = parts[1].parse()?;
                let y: f32 = parts[2].parse()?;
                let width: f32 = parts[3].parse()?;
                let height: f32 = parts[4].parse()?;
                
                boxes.push(BoundingBox::new(x, y, width, height, class_id));
            }
        }

        Ok(boxes)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&YoloItem> {
        self.items.get(index)
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}
