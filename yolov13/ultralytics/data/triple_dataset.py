# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.ops import resample_segments
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .utils import (
    HELP_URL,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)
from ultralytics.utils import LOGGER

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"


class TripleYOLODataset(BaseDataset):
    """
    Dataset class for loading object detection with triple image inputs.
    
    Expected directory structure:
    dataset/
    ├── images/
    │   ├── primary/     # Images with labels (first input)
    │   ├── detail1/     # Additional detail images (second input)
    │   └── detail2/     # Additional detail images (third input)
    └── labels/          # Label files corresponding to primary images
    
    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the TripleYOLODataset with triple image configuration."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)
        
        # Set up paths for triple images
        self.setup_triple_paths()

    def setup_triple_paths(self):
        """Setup paths for triple image inputs."""
        # Assume primary images are in img_path, derive other paths
        img_path = Path(self.img_path)
        
        if 'primary' in str(img_path):
            # If already in primary folder
            base_path = img_path.parent
            self.primary_path = img_path
            self.detail1_path = base_path / 'detail1'
            self.detail2_path = base_path / 'detail2'
        else:
            # Assume img_path is base images folder
            self.primary_path = img_path / 'primary'
            self.detail1_path = img_path / 'detail1'
            self.detail2_path = img_path / 'detail2'
        
        # Get corresponding file lists
        self.primary_files = self.get_img_files(str(self.primary_path))
        self.detail1_files = self.get_corresponding_files(self.primary_files, self.detail1_path)
        self.detail2_files = self.get_corresponding_files(self.primary_files, self.detail2_path)
        
        # Update im_files to primary files for label matching
        self.im_files = self.primary_files

    def get_corresponding_files(self, primary_files, detail_path):
        """Get corresponding detail files for primary files."""
        detail_files = []
        for primary_file in primary_files:
            # Get the filename and find corresponding file in detail folder
            filename = Path(primary_file).name
            detail_file = detail_path / filename
            
            # If exact match doesn't exist, try to find similar file
            if not detail_file.exists():
                # Try different extensions or naming patterns
                stem = Path(primary_file).stem
                possible_files = list(detail_path.glob(f"{stem}.*"))
                if possible_files:
                    detail_file = possible_files[0]
                else:
                    # Fallback: use the primary image if detail is missing
                    detail_file = primary_file
                    LOGGER.warning(f"Detail image not found for {primary_file}, using primary image as fallback")
            
            detail_files.append(str(detail_file))
        
        return detail_files

    def load_image(self, i):
        """Load triple images for training."""
        # Load primary image (with labels)
        primary_path = self.primary_files[i]
        primary_img = cv2.imread(primary_path)
        
        # Load detail images
        detail1_path = self.detail1_files[i]
        detail2_path = self.detail2_files[i]
        
        detail1_img = cv2.imread(detail1_path)
        detail2_img = cv2.imread(detail2_path)
        
        # Ensure all images have the same size
        if primary_img is not None:
            h, w = primary_img.shape[:2]
            
            if detail1_img is not None:
                detail1_img = cv2.resize(detail1_img, (w, h))
            else:
                detail1_img = primary_img.copy()
                
            if detail2_img is not None:
                detail2_img = cv2.resize(detail2_img, (w, h))
            else:
                detail2_img = primary_img.copy()
        
        # Convert BGR to RGB
        if primary_img is not None:
            primary_img = cv2.cvtColor(primary_img, cv2.COLOR_BGR2RGB)
        if detail1_img is not None:
            detail1_img = cv2.cvtColor(detail1_img, cv2.COLOR_BGR2RGB)
        if detail2_img is not None:
            detail2_img = cv2.cvtColor(detail2_img, cv2.COLOR_BGR2RGB)
        
        return [primary_img, detail1_img, detail2_img]

    def __getitem__(self, index):
        """Returns transformed triple image and target information at given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        """Get triple images and labels."""
        label = self.get_label_info(index)
        label["img"] = self.load_image(index)
        label["ratio_pad"] = (1.0, (0.0, 0.0))  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        label = self.update_labels_info(label)
        return label

    def update_labels_info(self, label):
        """Custom transforms for triple image inputs."""
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # Update if keypoints or segments exist  
        if len(segments) > 0:
            # list of segments to segments array, (N, 1000, 2)
            segments = np.stack(segments, axis=0)
        else:
            segments = np.zeros((0, 1000, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    def collate_fn(self, batch):
        """Collate function for triple image batches."""
        new_batch = []
        for item in batch:
            # Handle triple image inputs
            if isinstance(item["img"], list) and len(item["img"]) == 3:
                # Keep the structure for triple inputs
                new_item = {key: value for key, value in item.items()}
                new_batch.append(new_item)
            else:
                new_batch.append(item)
        
        return new_batch


def build_triple_transforms(cfg, dataset, hyp):
    """Builds transforms for triple image datasets."""
    pre_transform = Compose([
        TripleLetterBox(new_shape=(cfg.imgsz, cfg.imgsz), auto=False, scaleFill=hyp.mosaic == 0)
    ])
    
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints
    if dataset.use_keypoints:
        keypoint_kpt_shape = dataset.data.get("kpt_shape", None)
        if keypoint_kpt_shape is None:
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        pre_transform += [RandomLoadText(max_samples=min(dataset.ni, 1000) if cfg.save_json else 0)]

    if hyp.mosaic > 0:
        # TODO: Add triple mosaic support
        pass

    transforms = v8_transforms(dataset, cfg.imgsz, hyp, stretch=hyp.stretch)
    transforms.insert(-1, pre_transform)
    transforms = Compose(transforms)
    
    return transforms


class TripleLetterBox:
    """
    Resize triple images and pad for detection, instance segmentation and pose.
    """

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """Initialize TripleLetterBox for resizing triple images."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center

    def __call__(self, labels=None, image=None):
        """Return updated labels and triple image with added border."""
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img[0].shape[:2] if isinstance(img, list) else img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            if isinstance(img, list):
                # Handle triple images
                img = [cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR) for im in img]
            else:
                img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        
        if isinstance(img, list):
            # Apply padding to all triple images
            img = [cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)) 
                   for im in img]
        else:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels for triple image letterboxing."""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"][0].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels