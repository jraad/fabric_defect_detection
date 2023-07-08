import os

import numpy as np
import cv2

import torch
import torchvision.transforms as transforms

from models import BinaryClassifier

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

def patch_image(img: np.array) -> list:
    """Convert full fabric image to patches.
    
    This method also applies some light preprocessing (hist equalization)
    Input:
        img (np.array): np representation of image
    Output:
        patches (list[np.array]): 256x256 patches of full images
    """
    img_new = cv2.resize(img, (4096, 256)) 
    img_new = cv2.equalizeHist(img_new) / 255.
    
    return [torch.Tensor(img_new[:,i:i+256]).reshape((1, 256, 256)) for i in range(0, 4096, 256)]

def detect_defects(img: np.array):
    patches = patch_image(img)
    predictions = []
    for patch in patches:
        res = model(transform(patch).reshape(1, 1, 224, 224).to(device))
        predictions.append(int(res.cpu().detach() >= 0.5))
    
    return sum(predictions)

    