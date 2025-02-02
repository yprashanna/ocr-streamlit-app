import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import os

# OCR functions with preprocessing
def preprocess_image(image):
    # Convert image to grayscale
    gray_image = ImageOps.grayscale(image)
    
    # Apply thresholding to improve text visibility
    np_image = np.array(gray_image)
    _, thresholded = cv2.threshold(np_image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to PIL Image
    processed_image = Image.fromarray(thresholded)
    return processed_image

def ocr_image(image):
    processed_image = preprocess_image(image)
    text = pytesseract.image_to_string(processed_image, config='--psm 6 --oem 3')
    return text

def ocr_pdf(pdf_file):
    images = convert_from_path(pdf_file)
    text = ''
    for image in images:
        processed_image = preprocess_image(image)
        text += pytesseract.image_to_string(processed_image, config='--psm 6 --oem 3')
    return text

# Streamlit App
st.title("OCR App for Images and PDFs (Supports Handwritten Notes)")
st.write("Upload an image or a PDF to extract text, including handwritten notes.")

uploaded_file = st.file_uploader("Choose an Image or PDF", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    
    if 'image' in file_type:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        text = ocr_image(image)
        st.subheader("Extracted Text:")
        st.text_area("", text, height=200)
    
    elif 'pdf' in file_type:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        text = ocr_pdf("temp.pdf")
        os.remove("temp.pdf")
        st.subheader("Extracted Text:")
        st.text_area("", text, height=200)
