 
import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import os

# Set Tesseract CMD path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# OCR functions
def ocr_image(image):
    text = pytesseract.image_to_string(image)
    return text

def ocr_pdf(pdf_file):
    images = convert_from_path(pdf_file)
    text = ''
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

# Streamlit App
st.title("OCR App for Images and PDFs")
st.write("Upload an image or a PDF to extract text.")

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
