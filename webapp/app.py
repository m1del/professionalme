import gradio as gr
import numpy as np
import torch
from PIL import Image

from diffusers import AutoPipelineForImage2Image

# Load model
pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0")
count = 0

def generate_photo(input_image):
    global count
    count += 1

    # Prompts for Stable Diffusion
    prompt = "Professional LinkedIn photo, person in a suit, realistic skin texture, photorealistic, hyper realism, 85mm portrait photography, hard rim lighting photography, centered"
    negative_prompt = "deformed, disfigured, poor details, bad anatomy, change person, change age, change race, change hair, change face"

    # Process input image
    input_image = Image.open(input_image).convert("RGB")

    # Generate output image
    images = pipe(prompt=prompt, negative_prompt=negative_prompt, image=input_image, strength=0.75, guidance_scale=7.5).images
    images[0].save(f"../results/{count}.png")

    return images[0]

# Interface
demo = gr.Interface(
    fn=generate_photo,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Image(),
    live=True,
    title="ProfessionalMe",
    description="Upload a selfie.",
)

# Launch app
demo.launch(share=False)
