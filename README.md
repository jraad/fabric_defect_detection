# Fabric Defect Detection
This repository stores code for training and deploying models for detecting defects in the AITEX Fabric Image Dataset. It contains routines to train multiple model versions as well as a Streamlit UI for inference (classification and segmentation).

## Setup

## Modeling Approach

## UI
The Streamlit UI allows an upload of a .png fabric image. This image is processed and each patch is run through the binary classification algorithm. If a defect is detected in any patch, the corresponing patches are run through the segmentation algorithm. The algorithm returns an image with a border around the defective patch in red, and the defect itself highlighted in green, as below. If no defect is detected, only one image is display.

![Alt text](ui.png)

## Limitations and Future Work