import streamlit as st
import numpy as np
from io import BytesIO

import torch
from PIL import Image

from diffusers import AutoPipelineForImage2Image

if 'count' not in st.session_state:
    st.session_state.count = 0

# Generate photo
def generate_photo(input_image):
    st.session_state.count += 1

    prompt = "Professional LinkedIn photo, person in a suit, realistic skin texture, photorealistic, hyper realism, 85mm portrait photography, hard rim lighting photography, centered"
    negative_prompt = "deformed, disfigured, poor details, bad anatomy, change person, change age, change race, change hair, change face"

    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0")

    images = pipe(prompt=prompt, negative_prompt=negative_prompt, image=input_image, strength=0.75, guidance_scale=7.5).images
    images[0].save(f"../results/{st.session_state.count}.png")

    return images[0]


def main():
    # Page configurations
    st.set_page_config(page_title="ProfessionalMe", page_icon="👨‍💼")
    st.title("ProfessionalMe")

    # Prompt for photo upload
    uploaded_file = st.file_uploader("Upload a selfie", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = BytesIO(uploaded_file.read())
        input_image = Image.open(file_bytes).convert("RGB")

        # Generate professional photo
        generated_image = generate_photo(input_image)

        # Display photo
        st.image(generated_image, caption="Generated Professional Photo", use_column_width=True)


if __name__ == "__main__":
    main()