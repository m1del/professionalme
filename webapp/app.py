import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2


def generate_photo(image, prompt):
    # Convert the image to a format that can be sent via HTTP
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = buffered.getvalue()

    # Define the URL of the Flask server
    url = "http://localhost:5000/transform-image"  # Adjust this to your Flask server's address

    # Send the image and prompt to the Flask server
    response = requests.post(url, json={"image_data": img_str, "prompt": prompt})

    # Convert the response back into an image
    received_image = Image.open(BytesIO(response.content))
    return received_image


def main():
    st.title("ProfessionalMe")

    uploaded_file = st.file_uploader("Upload a selfie", type=["jpg", "jpeg", "png"])
    prompt = st.text_input(
        "Enter a description for the transformation (e.g., 'A professional headshot')"
    )

    if uploaded_file is not None and prompt:
        # Read the uploaded image
        file_bytes = BytesIO(uploaded_file.read())
        input_image = Image.open(file_bytes)

        # Generate professional photo
        generated_image = generate_photo(input_image, prompt)

        # Display photo
        st.image(
            generated_image,
            caption="Generated Professional Photo",
            use_column_width=True,
        )
    else:
        st.write("Please upload an image and enter a prompt.")


if __name__ == "__main__":
    main()
