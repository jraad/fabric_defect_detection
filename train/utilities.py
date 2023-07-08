import os
import glob

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

class AITEX(Dataset):
    """
    Builds dataset in memory for AITEX fabric defects.
    
    Note: this could just as easily be done with data streaming but not really necessary here.
    """
    def __init__(self, file_dir, greyscale=True):
        """Read images."""
        self.image_dims = (256, 4096)
        self.file_dir = file_dir
        self.normal_dir = os.path.join(self.file_dir, "NODefect_images")
        self.defect_dir = os.path.join(self.file_dir, "Defect_images")
        self.mask_dir = os.path.join(self.file_dir, "Mask_images")
        
        # get image names
        self.normal_images = [x for x in glob.glob(os.path.join(self.normal_dir, "**"), recursive=True) if x.endswith(".png")]
        self.defect_masks = [x for x in glob.glob(os.path.join(self.mask_dir, "**"), recursive=True) if x.endswith("mask.png")]
        self.mask_roots = [x.split("\\")[-1].split("_mask.png")[0] for x in self.defect_masks]
        self.defect_images = [os.path.join(self.defect_dir, f"{x}.png") for x in self.mask_roots]

        # validate defect_images all exist
        for x in self.defect_masks:
            if not os.path.exists(x):
                raise Exception("Could not find correct defect image")
        
        # load images
        self.image_paths = self.normal_images.copy()
        self.image_paths.extend(self.defect_images)
        self.images = [cv2.imread(x, cv2.IMREAD_GRAYSCALE if greyscale else cv2.IMREAD_COLOR) for x in self.image_paths]        

        # generate blank masks for normal images
        self.masks = [np.zeros(self.image_dims, dtype=np.uint8) for x in self.normal_images]
        # self.masks.extend(self.defect_masks)
        self.masks.extend(
            [cv2.threshold(cv2.imread(x, cv2.IMREAD_GRAYSCALE), 0, 1, cv2.THRESH_BINARY)[1] for x in self.defect_masks]
        )

        # get class labels for all images
        self.classes = [int(x.split("\\")[-1].split("_")[1]) for x in self.image_paths]

        # get fabric types for all images
        self.fabric_types = [int(x.split("\\")[-1].split("_")[2].split(".")[0]) for x in self.image_paths]
        self.fabric_ref = np.unique(self.fabric_types)

    def __len__(self):
        """Get length of full dataset."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """Return specific index of dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.images[idx], self.masks[idx], self.classes[idx], self.image_paths[idx], self.fabric_types[idx]
    
    def get_by_type(self, fabric_type):
        """Return all images for a given fabric type."""
        indices = [x for x, y in enumerate(self.fabric_types) if y==fabric_type]

        filtered_images = [self.images[x] for x in indices]
        filtered_masks = [self.masks[x] for x in indices]
        filtered_classes = [self.classes[x] for x in indices]
        filtered_paths = [self.image_paths[x] for x in indices]

        return filtered_images, filtered_masks, filtered_classes, filtered_pathslocals
    
class AITEXPatched(AITEX):
    """Extension of previously defined object for patching to 256^2."""
    def __init__(self, *args, transform, **kwargs,):
        super(AITEXPatched, self).__init__(*args, **kwargs)

        self.num_classes = 2
        self.class_ref = {0: "normal", 1: "defect"}

        self.transform = transform
        self.patched_images = []
        self.patched_masks = []
        self.has_defect = []
        self.patched_labels = []
        for index, img in enumerate(self.images):
            img_new = cv2.resize(img, (4096, 256)) 
            img_new = cv2.equalizeHist(img_new) / 255.
            self.patched_images.extend([torch.Tensor(img_new[:,i:i+256]).reshape((1, 256, 256)) for i in range(0, 4096, 256)])

            mask_new = cv2.resize(self.masks[index], (4096, 256))
            mask_patches = [torch.Tensor(mask_new[:,i:i+256]).reshape((1, 256, 256)) for i in range(0, 4096, 256)]
            self.patched_masks.extend(mask_patches)

            self.has_defect.extend([1 if torch.sum(x) > 0 else 0 for x in mask_patches])

            true_label = self.classes[index]
            self.patched_labels.extend([true_label if torch.sum(x) > 0 else 0 for x in mask_patches])

    def __len__(self):
        """Get length of full dataset."""
        return len(self.patched_images)    
    
    def __getitem__(self, idx):
        """Return specific index of dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.transform(self.patched_images[idx]), self.has_defect[idx]
    
class AITEXPatchedSegmentation(AITEXPatched):
    """Extension of previously defined object for patching to 256^2."""
    def __init__(self, *args, **kwargs,):
        super(AITEXPatchedSegmentation, self).__init__(*args, **kwargs)

        self.defect_indices = [x for x,y in enumerate(self.has_defect) if y==1]
        self.defect_images = [self.patched_images[x] for x in self.defect_indices]
        self.defect_masks = [self.patched_masks[x] for x in self.defect_indices]
        self.defect_labels = [self.patched_labels[x] for x in self.defect_indices]
    
    def __len__(self):
        """Get length of full dataset."""
        return len(self.defect_images)    
    
    def __getitem__(self, idx):
        """Return specific index of dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.transform(self.defect_images[idx]), self.transform(self.defect_masks[idx]) #, self.defect_labels[idx]
    