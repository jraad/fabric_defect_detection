from gdown import download
import os

models_to_download = [
    {
        "url": "https://drive.google.com/uc?id=1DAGs5eG-jTSvWMNUVMPGkSVfBO_kfb0G",
        "name": "unet_seg_200epoch.pt"
    }
]

model_dir = "models"

for model_info in models_to_download:
    download(model_info["url"], os.path.join(model_dir, model_info["name"]), quiet=False)