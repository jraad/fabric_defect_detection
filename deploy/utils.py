import os

import numpy as np
import cv2

import torch
import torchvision.transforms as transforms

from models import BinaryClassifier, UNet

# Some useful paths
root = os.path.abspath(os.path.join(os.getcwd(), "."))
model_dir = os.path.join(root, "models")
data_dir = os.path.join(root, "data")
aitex_dir = os.path.join(data_dir, "aitex")

# General torch stuff
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.Resize((224, 224))
])

# Load binary classifier for patches
model_path = os.path.join(model_dir, "bigger_binary_F1_0.98.pth")
model = BinaryClassifier()
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Load UNet for defect segmentation
model_path = os.path.join(model_dir, "unet_seg_200epoch.pt")
seg_model = UNet()
seg_model.load_state_dict(torch.load(model_path))
seg_model.to(device)
seg_model.eval()


def bytes_to_np(img_bytes: bytes) -> np.array:
    """Decode bytes into numpy array.
    
    Inputs:
        img_bytes (bytes): byte representation of a png file
    Returns:
        img (np.array): decoded image for downstream use
    """
    file_bytes = np.asarray(bytearray(img_bytes.read()), dtype=np.uint8)
    img  = cv2.imdecode(file_bytes, 1)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def patch_image(img: np.array, equalize=True) -> list:
    """Convert full fabric image to patches.
    
    This method also applies some light preprocessing (hist equalization)
    Input:
        img (np.array): np representation of image
    Output:
        patches (list[np.array]): 256x256 patches of full images
    """
    img_new = cv2.resize(img, (4096, 256))
    if equalize:
        img_new = cv2.equalizeHist(img_new) / 255.
    
    return [torch.Tensor(img_new[:,i:i+256]).reshape((1, 256, 256)) for i in range(0, 4096, 256)]


def classify_image(img: np.array):
    """Classifies each patch of an image as defective or not defective.
    
    Input:
        img (np.array): np representation of image
    Output:
        predictions (list): int list of prediction for each patch
        max_res (float): max probability of defect
    """
    patches = patch_image(img)
    predictions = []
    max_res = 0
    for patch in patches:
        res = model(transform(patch).reshape(1, 1, 224, 224).to(device))
        if res.item() > max_res:
            max_res = res.item()
        predictions.append(int(res.cpu().detach() >= 0.5))
    
    return predictions, max_res


def get_defect_segment(img: np.array, predictions: list) -> np.array:
    """Segments defects from patches that had a defect detected.
    
    Input:
        img (np.array): np representation of image
        predictions (list): list of integers indicating whether specific patch was defective
    Output:
        segmented (np.array): original image with segment overlay
    """
    patches = patch_image(img)
    original_patches = patch_image(img, equalize=False)
    original_patches = [cv2.cvtColor(x.numpy().reshape(256, 256), cv2.COLOR_GRAY2BGR) / 255. for x in original_patches]
    for idx, has_defect in enumerate(predictions):
        if has_defect:
            # Get prediction
            outputs = seg_model(patches[idx].reshape(1, 1, 256, 256).to(device))

            # Normalize and threshold prediction mask
            mask = outputs.cpu().detach().numpy().reshape(256, 256)
            amin = np.amin(mask)
            amax = np.amax(mask)
            mask = (mask - amin) / (amax - amin)
            _, thresh = cv2.threshold(mask, 0.75, 1, cv2.THRESH_BINARY)
            threshed_mask = thresh.copy().astype(np.uint8)
            thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            thresh[np.all(thresh == (1, 1, 1), axis=-1)] = (1, 0, 0)

            # Add mask to original image and include rectangle around defective patch for clarity
            masked_img = original_patches[idx].copy()
            masked_img[threshed_mask == 1] = (0, 1, 0)
            masked_img = cv2.rectangle(masked_img, (0, 0), (256, 256), (1, 0, 0), 6)
            original_patches[idx] = masked_img

    return np.concatenate(original_patches, axis=1) # / 255.
