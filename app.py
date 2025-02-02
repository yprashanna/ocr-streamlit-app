import streamlit as st
from PIL import Image, ImageOps
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import os
from transformers import pipeline

# Load NLP model for post-processing
nlp = pipeline("text2text-generation", model="t5-small")

# Custom CSS for sleek UI
st.markdown("""
    <style>
        .chat-container {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            max-height: 500px;
            overflow-y: auto;
        }
        .chat-input {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: white;
            padding: 10px 20px;
            box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
        }
        .uploaded-image {
            max-width: 100%;
            border-radius: 10px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# OCR functions with preprocessing
def preprocess_image(image):
    gray_image = ImageOps.grayscale(image)
    np_image = np.array(gray_image)
    thresholded = np.where(np_image > 150, 255, 0).astype(np.uint8)
    processed_image = Image.fromarray(thresholded)
    return processed_image

def ocr_image(image):
    processed_image = preprocess_image(image)
    raw_text = pytesseract.image_to_string(processed_image, config='--psm 6 --oem 3')
    refined_text = nlp(raw_text, max_length=500, do_sample=False)[0]['generated_text']
    return refined_text

def ocr_pdf(pdf_file):
    images = convert_from_path(pdf_file)
    text = ''
    for image in images:
        processed_image = preprocess_image(image)
        raw_text = pytesseract.image_to_string(processed_image, config='--psm 6 --oem 3')
        refined_text = nlp(raw_text, max_length=500, do_sample=False)[0]['generated_text']
        text += refined_text + "\n"
    return text

# Streamlit App UI
st.title("📄 AI-Powered OCR App")
st.write("Upload an image or PDF to extract text, including handwritten notes and mathematical equations.")

# Chat-like display container
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg", "pdf"], label_visibility='collapsed')

    if uploaded_file is not None:
        file_type = uploaded_file.type

        if 'image' in file_type:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True, output_format="JPEG")
            text = ocr_image(image)
            st.write(f"**Extracted Text:**\n{text}")

        elif 'pdf' in file_type:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            text = ocr_pdf("temp.pdf")
            os.remove("temp.pdf")
            st.write(f"**Extracted Text:**\n{text}")

    st.markdown('</div>', unsafe_allow_html=True)

# Chat-like input at the bottom
st.markdown('<div class="chat-input">', unsafe_allow_html=True)
custom_prompt = st.text_input("Type your message or upload a file above...", key="chat_input")
st.markdown('</div>', unsafe_allow_html=True)

if custom_prompt:
    st.write(f"**You:** {custom_prompt}")
