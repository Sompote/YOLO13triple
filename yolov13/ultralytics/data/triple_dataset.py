# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import random
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
from ultralytics.utils.ops import resample_segments, xyxyxyxy2xywhr
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
from .dataset import YOLODataset
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


class TripleFormat(Format):
    """
    Format class for triple image datasets.
    Handles conversion of triple image lists to tensors.
    """
    
    def __call__(self, labels):
        """
        Formats image annotations for triple image object detection.
        Handles the case where img is a list of 3 images.
        """
        img = labels.pop("img")
        
        # Extract height and width - handle both single image and triple image cases
        if isinstance(img, list) and len(img) == 3:
            h, w = img[0].shape[:2]  # Use first image for shape reference
        else:
            h, w = img.shape[:2]
        
        cls = labels.pop("cls")
        instances = labels.pop("instances")
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)

        if self.return_mask:
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks)
            else:
                # For triple images, use first image for mask dimensions
                ref_img = img[0] if isinstance(img, list) else img
                masks = torch.zeros(
                    1 if self.mask_overlap else nl, ref_img.shape[0] // self.mask_ratio, ref_img.shape[1] // self.mask_ratio
                )
            labels["masks"] = masks
            
        labels["img"] = self._format_img(img)
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        
        if self.return_keypoint:
            labels["keypoints"] = torch.from_numpy(instances.keypoints)
            if self.normalize:
                labels["keypoints"][..., 0] /= w
                labels["keypoints"][..., 1] /= h
                
        if self.return_obb:
            labels["bboxes"] = (
                xyxyxyxy2xywhr(torch.from_numpy(instances.segments)) if len(instances.segments) else torch.zeros((0, 5))
            )
            
        # NOTE: need to normalize obb in xywhr format for width-height consistency
        if self.normalize:
            labels["bboxes"][:, [0, 2]] /= w
            labels["bboxes"][:, [1, 3]] /= h
            
        # Then we can use collate_fn
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
            
        return labels
    
    def _format_img(self, img):
        """
        Formats triple images from list of Numpy arrays to a single PyTorch tensor.
        
        Args:
            img: Either a single numpy array or list of 3 numpy arrays (triple images)
            
        Returns:
            torch.Tensor: Formatted image tensor with shape (C*3, H, W) for triple images
                         or (C, H, W) for single images
        """
        if isinstance(img, list) and len(img) == 3:
            # Handle triple images
            formatted_imgs = []
            for single_img in img:
                if len(single_img.shape) < 3:
                    single_img = np.expand_dims(single_img, -1)
                single_img = single_img.transpose(2, 0, 1)
                single_img = np.ascontiguousarray(
                    single_img[::-1] if random.uniform(0, 1) > self.bgr else single_img
                )
                formatted_imgs.append(single_img)
            
            # Concatenate along channel dimension (C*3, H, W)
            img_tensor = torch.from_numpy(np.concatenate(formatted_imgs, axis=0))
            return img_tensor
        else:
            # Handle single image (fallback to parent method)
            return super()._format_img(img)


class TripleYOLODataset(YOLODataset):
    """
    Dataset class for loading object detection with triple image inputs.
    
    Expected directory structure:
    dataset/
    ├── images/
    │   ├── primary/     # Images with labels (first input) - REQUIRED
    │   ├── detail1/     # Additional detail images (second input) - OPTIONAL
    │   └── detail2/     # Additional detail images (third input) - OPTIONAL
    └── labels/
        └── primary/     # Label files for primary images ONLY - detail1/detail2 don't need labels
    
    Note: Only primary images require labels. Detail1 and detail2 images are additional
    context inputs. If detail images are missing, primary images are used as fallback.
    
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
        
        # Ensure data is passed correctly to parent class
        if data is not None:
            kwargs['data'] = data
        kwargs['task'] = task
        
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
                    # Fallback: use the primary image if detail is missing (expected behavior)
                    detail_file = primary_file
            
            detail_files.append(str(detail_file))
        
        return detail_files

    def load_image(self, i):
        """Load triple images for training."""
        # Load primary image (with labels)
        primary_path = self.primary_files[i]
        primary_img = cv2.imread(primary_path)
        
        if primary_img is None:
            raise FileNotFoundError(f"Primary image not found: {primary_path}")
        
        # Store original shape
        ori_shape = primary_img.shape[:2]  # HW
        
        # Load detail images
        detail1_path = self.detail1_files[i]
        detail2_path = self.detail2_files[i]
        
        detail1_img = cv2.imread(detail1_path)
        detail2_img = cv2.imread(detail2_path)
        
        # Ensure all images have the same size
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
        primary_img = cv2.cvtColor(primary_img, cv2.COLOR_BGR2RGB)
        detail1_img = cv2.cvtColor(detail1_img, cv2.COLOR_BGR2RGB)  
        detail2_img = cv2.cvtColor(detail2_img, cv2.COLOR_BGR2RGB)
        
        triple_images = [primary_img, detail1_img, detail2_img]
        resized_shape = ori_shape  # No resizing yet, will be done by transforms
        
        return triple_images, ori_shape, resized_shape

    def __getitem__(self, index):
        """Returns transformed triple image and target information at given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        """Get triple images and labels."""
        # Use base class approach for getting labels
        from copy import deepcopy
        label = deepcopy(self.labels[index])
        label.pop("shape", None)  # shape is for rect, remove it
        
        # Load triple images instead of single image
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        # Note: ratio_pad will be set correctly by TripleLetterBox in transforms
        
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        label = self.update_labels_info(label)
        return label

    def build_transforms(self, hyp=None):
        """Builds transforms with TripleFormat for triple image datasets."""
        # Use minimal transforms for triple images
        # TODO: Implement triple-image-compatible augmentations
        # Current augmentations (RandomHSV, RandomFlip, Albumentations, etc.) 
        # expect single images, not lists of 3 images
        transforms = Compose([TripleLetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        
        transforms.append(
            TripleFormat(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=0.0,  # Disable BGR flipping until triple-compatible version is implemented
            )
        )
        return transforms

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

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches for triple image datasets."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                # Handle both regular tensors and triple image tensors
                try:
                    value = torch.stack(value, 0)
                except Exception as e:
                    # Convert any remaining lists to tensors
                    converted_value = []
                    for v in value:
                        if isinstance(v, list):
                            raise ValueError("Images are still lists after formatting - check TripleFormat implementation")
                        converted_value.append(v)
                    value = torch.stack(converted_value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
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
        
        # Set ratio_pad in the correct format for validation
        # Format: (ratio_values, (pad_left, pad_top))
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (padw, padh))
        else:
            labels["ratio_pad"] = (ratio, (padw, padh))
        
        return labels