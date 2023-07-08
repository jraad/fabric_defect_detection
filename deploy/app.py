import streamlit as st

from utils import bytes_to_np, detect_defects


# image ingestion
image_input = st.file_uploader("Upload a fabric image to initiate defect detection:", type="png")
if image_input is not None:
    img = bytes_to_np(image_input)

    st.image(img)

    result = detect_defects(img)
    if result:
        st.write("Defect detected. Obtaining segmentation mask.")
    else:
        st.write("No defects detected!")